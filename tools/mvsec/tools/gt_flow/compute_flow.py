""" Computes optical flow from two poses and depth images """

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.linalg import logm
import cv2
try:
    from quaternion import quaternion
except ImportError:
    class quaternion:
        def __init__(self,w,x,y,z):
            self.w = w
            self.x = x
            self.y = y
            self.z = z
    
        def norm(self):
            return self.w**2 + self.x**2 + self.y**2 + self.z**2
    
        def inverse(self):
            qnorm = self.norm()
            return quaternion(self.w/qnorm,
                              -self.x/qnorm,
                              -self.y/qnorm,
                              -self.z/qnorm)
    
        def __mul__(q1, q2):
            r = quaternion(q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z,
                           q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y,
                           q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
                           q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w)
            return r
    
        def __rmul__(q1, s):
            return quaternion(q1.w*s, q1.x*s, q1.y*s, q1.z*s)
    
        def __sub__(q1, q2):
            r = quaternion(q1.w-q2.w,
                           q1.x-q2.x,
                           q1.y-q2.y,
                           q1.z-q2.z)
            return r
    
        def __div__(q1, s):
            return quaternion(q1.w/s, q1.x/s, q1.y/s, q1.z/s)

class Flow:
    """
    - parameters
        - calibration :: a Calibration object from calibration.py
    """
    def __init__(self, calibration):
        self.cal = calibration
        self.Pfx = self.cal.intrinsic_extrinsic['cam0']['projection_matrix'][0][0]
        self.Ppx = self.cal.intrinsic_extrinsic['cam0']['projection_matrix'][0][2]
        self.Pfy = self.cal.intrinsic_extrinsic['cam0']['projection_matrix'][1][1]
        self.Ppy = self.cal.intrinsic_extrinsic['cam0']['projection_matrix'][1][2]

        intrinsics = self.cal.intrinsic_extrinsic['cam0']['intrinsics']
        self.K = np.array([[intrinsics[0], 0., intrinsics[2]],
                           [0., intrinsics[1], intrinsics[3]],
                           [0., 0., 1.]])

        self.distortion_coeffs = np.array(self.cal.intrinsic_extrinsic['cam0']['distortion_coeffs'])
        resolution = self.cal.intrinsic_extrinsic['cam0']['resolution']

        # number of pixels in the camera
        #self.x_map = (self.cal.left_map[:,:,0]-self.Ppx)/self.Pfx
        #self.y_map = (self.cal.left_map[:,:,1]-self.Ppy)/self.Pfy
        #self.flat_x_map = self.x_map.ravel()
        #self.flat_y_map = self.y_map.ravel()

        # left_map takes into account the rectification matrix, which rotates the image.
        # For optical flow in the distorted image, this rotation needs to be removed.
        # In the end it's easier just to recompute the map.
        x_inds, y_inds = np.meshgrid(np.arange(resolution[0]),
                                     np.arange(resolution[1]))
        x_inds = x_inds.astype(np.float32)
        y_inds = y_inds.astype(np.float32)
        flat_x_inds = x_inds.reshape((-1))
        flat_y_inds = y_inds.reshape((-1))     
        points = np.stack((flat_x_inds[:, np.newaxis], flat_y_inds[:, np.newaxis]), axis=2)
        import cv2
        undistorted_points = cv2.fisheye.undistortPoints(points, self.K, self.distortion_coeffs)
        self.flat_x_map = np.squeeze(undistorted_points[:, :, 0])
        self.flat_y_map = np.squeeze(undistorted_points[:, :, 1])

        N = self.flat_x_map.shape[0]

        self.omega_mat = np.zeros((N,2,3))

        self.omega_mat[:,0,0] = self.flat_x_map * self.flat_y_map
        self.omega_mat[:,1,0] = 1+ np.square(self.flat_y_map)

        self.omega_mat[:,0,1] = -(1+np.square(self.flat_x_map))
        self.omega_mat[:,1,1] = -(self.flat_x_map*self.flat_y_map)

        self.omega_mat[:,0,2] = self.flat_y_map
        self.omega_mat[:,1,2] = -self.flat_x_map

        self.hsv_buffer = None

    def compute_flow_single_frame(self, V, Omega, depth_image, dt):
        """
        params:
            V : [3,1]
            Omega : [3,1]
            depth_image : [m,n]
        """
        flat_depth = depth_image.ravel()
        flat_depth[np.logical_or(np.isclose(flat_depth,0.0), flat_depth<0.)]
        mask = np.isfinite(flat_depth)

        fdm = 1./flat_depth[mask]
        fxm = self.flat_x_map[mask]
        fym = self.flat_y_map[mask]
        omm = self.omega_mat[mask,:,:]

        x_flow_out = np.zeros((depth_image.shape[0], depth_image.shape[1]))
        flat_x_flow_out = x_flow_out.reshape((-1))
        flat_x_flow_out[mask] = fdm * (fxm*V[2]-V[0])
        flat_x_flow_out[mask] +=  np.squeeze(np.dot(omm[:,0,:], Omega))

        y_flow_out = np.zeros((depth_image.shape[0], depth_image.shape[1]))
        flat_y_flow_out = y_flow_out.reshape((-1))
        flat_y_flow_out[mask] = fdm * (fym*V[2]-V[1])
        flat_y_flow_out[mask] +=  np.squeeze(np.dot(omm[:,1,:], Omega))

        flat_x_flow_out *= dt
        flat_y_flow_out *= dt

        return x_flow_out, y_flow_out


        x_inds, y_inds = np.meshgrid(np.arange(depth_image.shape[1]),
                                     np.arange(depth_image.shape[0]))
        
        flat_x_shifted = self.flat_x_map + flat_x_flow_out
        flat_y_shifted = self.flat_y_map + flat_y_flow_out

        points_shifted = np.stack((flat_x_shifted[np.newaxis, :], flat_y_shifted[np.newaxis, :]),
                                  axis=2)

        distorted_points_shifted = cv2.fisheye.distortPoints(points_shifted, 
                                                               self.K, 
                                                               self.distortion_coeffs)
        
        distorted_x, distorted_y = np.meshgrid(np.arange(depth_image.shape[1]),
                                               np.arange(depth_image.shape[0]))

        flat_distorted_x = distorted_x.reshape((-1))
        flat_distorted_y = distorted_y.reshape((-1))

        new_x_pts = np.squeeze(distorted_points_shifted[:, :, 0])
        new_y_pts = np.squeeze(distorted_points_shifted[:, :, 1])

        distorted_x_flow_out = np.zeros((depth_image.shape[0], depth_image.shape[1]))
        flat_distorted_x_flow_out = distorted_x_flow_out.reshape((-1))
        flat_distorted_x_flow_out[mask] = new_x_pts[mask] - flat_distorted_x[mask]

        distorted_y_flow_out = np.zeros((depth_image.shape[0], depth_image.shape[1]))
        flat_distorted_y_flow_out = distorted_y_flow_out.reshape((-1))
        flat_distorted_y_flow_out[mask] = new_y_pts[mask] - flat_distorted_y[mask]

        """
        plt.quiver(flat_distorted_x[::100],
                   flat_distorted_y[::100],
                   flat_distorted_x_flow_out[::100],
                   flat_distorted_y_flow_out[::100])

        plt.show()
        """

        return distorted_x_flow_out, distorted_y_flow_out
    
    def rot_mat_from_quaternion(self, q):
        R = np.array([[1-2*q.y**2-2*q.z**2, 2*q.x*q.y+2*q.w*q.z, 2*q.x*q.z-2*q.w*q.y],
                      [2*q.x*q.y-2*q.w*q.z, 1-2*q.x**2-2*q.z**2, 2*q.y*q.z+2*q.w*q.x],
                      [2*q.x*q.z+2*q.w*q.y, 2*q.y*q.z-2*q.w*q.x, 1-2*q.x**2-2*q.y**2]])
        return R

    def p_q_t_from_msg(self, msg):
        p = np.array([msg.pose.position.x,msg.pose.position.y,msg.pose.position.z])
        q = quaternion(msg.pose.orientation.w, msg.pose.orientation.x,
                            msg.pose.orientation.y, msg.pose.orientation.z)
        t = msg.header.stamp.to_sec()
        return p, q, t

    def compute_velocity_from_msg(self, P0, P1):
        p0, q0, t0 = self.p_q_t_from_msg(P0)
        p1, q1, t1 = self.p_q_t_from_msg(P1)

        # There's something wrong with the current function to go from quat to matrix.
        # Using the TF version instead.
        q0_ros = [q0.x, q0.y, q0.z, q0.w]
        q1_ros = [q1.x, q1.y, q1.z, q1.w]
        
        import tf
        H0 = tf.transformations.quaternion_matrix(q0_ros)
        H0[:3, 3] = p0

        H1 = tf.transformations.quaternion_matrix(q1_ros)
        H1[:3, 3] = p1

        # Let the homogeneous matrix handle the inversion etc. Guaranteed correctness.
        H01 = np.dot(np.linalg.inv(H0), H1)
        dt = t1 - t0

        V = H01[:3, 3] / dt
        w_hat = logm(H01[:3, :3]) / dt
        Omega = np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]])

        return V, Omega, dt

    def compute_velocity(self, p0, q0, p1, q1, dt):
        V = (p1-p0)/dt

        R_dot = ( self.rot_mat_from_quaternion(q1) - self.rot_mat_from_quaternion(q0) )/dt
        w_hat = np.dot(R_dot, self.rot_mat_from_quaternion(q1).T)

        Omega = np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]])

        return V, Omega

    def colorize_image(self, flow_x, flow_y):
        if self.hsv_buffer is None:
            self.hsv_buffer = np.empty((flow_x.shape[0], flow_x.shape[1],3))
            self.hsv_buffer[:,:,1] = 1.0
        self.hsv_buffer[:,:,0] = (np.arctan2(flow_y,flow_x)+np.pi)/(2.0*np.pi)

        self.hsv_buffer[:,:,2] = np.linalg.norm( np.stack((flow_x,flow_y), axis=0), axis=0 )

        self.hsv_buffer[:,:,2] = np.log(1.+self.hsv_buffer[:,:,2]) # hopefully better overall dynamic range in final video

        flat = self.hsv_buffer[:,:,2].reshape((-1))
        m = np.nanmax(flat[np.isfinite(flat)])
        if not np.isclose(m, 0.0):
            self.hsv_buffer[:,:,2] /= m

        return colors.hsv_to_rgb(self.hsv_buffer)

    def visualize_flow(self, flow_x, flow_y, fig):
        ax1 = fig.add_subplot(1,1,1)
        ax1.imshow( self.colorize_image(flow_x, flow_y) )


def dvs_img(shape, cloud):
    t0 = min(cloud[0].ts.to_sec(), cloud[-1].ts.to_sec())
    timg = np.zeros(shape, dtype=np.float)
    cimg = np.zeros(shape, dtype=np.float)
    pimg = np.zeros(shape, dtype=np.float)

    for e in cloud:
        if (e.y >= shape[0] or e.x >= shape[1]):
            continue
        #cimg[e.y, e.x] += 1
        timg[e.y, e.x] += (e.ts.to_sec() - t0)
        if (e.polarity):
            cimg[e.y, e.x] += 1
        else:
            pimg[e.y, e.x] += 1

    timg = np.divide(timg, cimg, out=np.zeros_like(timg), where=cimg!=0)
    return timg, cimg, pimg

def normalize_img(img):
    img[np.isnan(img)] = 0
    mn = np.sum(img) / np.count_nonzero(img)
    img *= 0.5 / mn
    return img

def overlay(img1, img2):
    return np.dstack((img1, img1, img2))

def undistort_img(img, K, dist):
    #K = np.array([[  226.38,     0.  ,   173.64],
    #              [    0.  ,   226.15,   133.73],
    #              [    0.  ,     0.  ,     1.  ]])

    # zero distortion coefficients work well for this image
    #D = np.array([-0.04803144, 0.01133096, -0.05537817,  0.02150097])
    D = dist
    # use Knew to scale the output
    Knew = K.copy()
    Knew[(0,1), (0,1)] = 0.87 * Knew[(0,1), (0,1)]

    img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
    return img_undistorted



global_delta = 0.05
global_scale_t = 20 * 255
global_scale_pn = 100
global_scale_pp = 100
global_shape = (200, 346)

def extract_all_events(data, flow, ename, eid):
    delta = global_delta
    base_name = "/home/ncos/Desktop/MVSEC/" + ename + '_' + str(eid) + "/train/"
    #shape = (260, 346)
    shape = global_shape

    n_event_msgs = len(data.left_cam_readers['/davis/left/events'])

    with open(base_name + 'cam.txt', 'w') as cam_txt:
        for row in flow.K:
            for num in row:
                cam_txt.write(str(num) + " ")
            cam_txt.write('\n')

    cloud = []
    cnt = 0
    for i in xrange(n_event_msgs):
        msg = data.left_cam_readers['/davis/left/events'][i].message
        for e in msg.events:
            cloud.append(e)

            if ((cloud[-1].ts - cloud[0].ts).to_sec() >= delta):
                print "Saving slice", cnt, "; message", i, "out of", n_event_msgs

                timg, cimg, pimg = dvs_img(shape, cloud)
                timg = undistort_img(timg, flow.K, flow.distortion_coeffs)
                cimg = undistort_img(cimg, flow.K, flow.distortion_coeffs)
                pimg = undistort_img(pimg, flow.K, flow.distortion_coeffs)

                time_name = base_name + 'time_' + str(cnt).rjust(10, '0') + ".png"
                count_name = base_name + 'cnt_' + str(cnt).rjust(10, '0') + ".png"
                polar_name = base_name + 'polarity_' + str(cnt).rjust(10, '0') + ".png"
                cmb_name = base_name + 'frame_' + str(cnt).rjust(10, '0') + ".png"

                #cv2.imwrite(time_name, 255 * timg)
                #cv2.imwrite(count_name, cimg)
                #cv2.imwrite(polar_name, pimg)

                cmb = np.dstack((cimg * global_scale_pp, timg * global_scale_t, pimg * global_scale_pn))
                cv2.imwrite(cmb_name, cmb)

                cloud = []
                cnt += 1


def extract_events_to_txt(data, flow, ename, eid):
    print "Saving all events to a .txt"

    delta = global_delta
    base_name = "/home/ncos/Desktop/MVSEC/" + ename + '_' + str(eid) + "/text/"
    #shape = (260, 346)
    shape = global_shape

    n_event_msgs = len(data.left_cam_readers['/davis/left/events'])

    with open(base_name + 'calib.txt', 'w') as cam_txt:
        for row in flow.K:
            for num in row:
                cam_txt.write(str(num) + " ")
            cam_txt.write('\n')


        cam_txt.write('\n')
        for num in flow.distortion_coeffs:
            cam_txt.write(str(num) + " ")
    
    if (n_event_msgs == 0):
        print "No event messages found!"

    e0 = data.left_cam_readers['/davis/left/events'][0].message.events[0]
    with open(base_name + 'events.txt', 'w') as e_txt:
        for i in xrange(n_event_msgs):
            msg = data.left_cam_readers['/davis/left/events'][i].message

            print "message", i, "out of", n_event_msgs
            for e in msg.events:
                x = int(e.x)
                y = int(e.y)
                ts = float((e.ts - e0.ts).to_sec())
                p = int(e.polarity)
                e_txt.write(str(ts) + " " + str(x) + " " + str(y) + " " + str(p) + "\n")


def experiment_flow(experiment_name, experiment_num, mode=0, start_ind=None, stop_ind=None):

    import time
    import calibration
    cal = calibration.Calibration(experiment_name)
    import ground_truth
    data = ground_truth.Dataset(experiment_name, experiment_num)

    flow = Flow(cal)
    P0 = None

    import downloader
    import os
    base_name = os.path.join(downloader.get_tmp(), experiment_name, experiment_name+str(experiment_num))

    if (mode == 0):
        extract_all_events(data, flow, experiment_name, experiment_num)
        return

    if (mode == 1):
        extract_events_to_txt(data, flow, experiment_name, experiment_num)
        return

    if experiment_name == "motorcycle":
        print "The motorcycle doesn't have lidar and we can't compute flow without it"
        return

    gt = ground_truth.GroundTruth(experiment_name, experiment_num)

    #depth_topic = '/davis/left/depth_image_raw'
    depth_topic = '/davis/left/depth_image_rect'
    #depth_topic = '/davis/left/depth_image_raw'
    nframes = len(gt.left_cam_readers[depth_topic])
    if stop_ind is not None:
        stop_ind = min(nframes, stop_ind)
    else:
        stop_ind = nframes

    if start_ind is not None:
        start_ind = max(0, start_ind)
    else:
        start_ind = 0

    nframes = stop_ind - start_ind


    depth_image, _ = gt.left_cam_readers[depth_topic](0)
    flow_shape = (nframes, depth_image.shape[0], depth_image.shape[1])
    x_flow_dist = np.zeros(flow_shape, dtype=np.float)
    y_flow_dist = np.zeros(flow_shape, dtype=np.float)
    depths      = np.zeros(flow_shape, dtype=np.float)
    timestamps = np.zeros((nframes,), dtype=np.float)
    Vs = np.zeros((nframes,3), dtype=np.float)
    Omegas = np.zeros((nframes,3), dtype=np.float)
    dTs = np.zeros((nframes,), dtype=np.float)

    ps = np.zeros((nframes,3), dtype=np.float)
    qs = np.zeros((nframes,4), dtype=np.float)

    sOmega = np.zeros((3,))
    sV = np.zeros((3,))

    base_name = "/home/ncos/Desktop/MVSEC/" + experiment_name + '_' + str(experiment_num)
    e0 = data.left_cam_readers['/davis/left/events'][0].message.events[0]

    print depth_image.shape
    print "Extracting velocity"
    for frame_num in range(nframes):
        P1 = gt.left_cam_readers['/davis/left/odometry'][frame_num+start_ind].message

        if P0 is not None:
            V, Omega, dt = flow.compute_velocity_from_msg(P0, P1)
            Vs[frame_num, :] = V
            Omegas[frame_num, :] = Omega
            dTs[frame_num] = dt

        timestamps[frame_num] = P1.header.stamp.to_sec()

        tmp_p, tmp_q, _ = flow.p_q_t_from_msg(P1)
        ps[frame_num, :] = tmp_p
        qs[frame_num, 0] = tmp_q.w
        qs[frame_num, 1] = tmp_q.x
        qs[frame_num, 2] = tmp_q.y
        qs[frame_num, 3] = tmp_q.z

        P0 = P1

    with open(base_name + '/text/cam.txt', 'w') as cam_txt:
        for row in flow.K:
            for num in row:
                cam_txt.write(str(num) + " ")
            cam_txt.write('\n')

    with open(base_name + '/text/poses.txt', 'w') as poses_txt:
        for frame_num in range(nframes):
            q = qs[frame_num]
            p = ps[frame_num]

            q_ = quaternion(q[0], q[1], q[2], q[3])
            Rflat = np.ravel(flow.rot_mat_from_quaternion(q_))
            for n in Rflat:
                poses_txt.write(str(n) + ' ')
            poses_txt.write(str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + '\n')

    filter_size = 10

    smoothed_Vs = Vs
    smoothed_Omegas = Omegas

    import cv2

    print "Computing flow"
    for frame_num in range(nframes):
        depth_image = gt.left_cam_readers[depth_topic][frame_num+start_ind]
        depth_image.acquire()

        if frame_num-filter_size < 0:
            V = np.mean(Vs[0:frame_num+filter_size+1,:],axis=0)
            Omega = np.mean(Omegas[0:frame_num+filter_size+1,:], axis=0)
        elif frame_num+filter_size >= nframes:
            V = np.mean(Vs[frame_num-filter_size:nframes,:],axis=0)
            Omega = np.mean(Omegas[frame_num-filter_size:nframes,:], axis=0)
        else:
            V = np.mean(Vs[frame_num-filter_size:frame_num+filter_size+1,:],axis=0)
            Omega = np.mean(Omegas[frame_num-filter_size:frame_num+filter_size+1,:], axis=0)
        dt = dTs[frame_num]

        smoothed_Vs[frame_num, :] = V
        smoothed_Omegas[frame_num, :] = Omega

        #x_flow_dist[frame_num,:,:] = flow_x_dist
        #y_flow_dist[frame_num,:,:] = flow_y_dist
        depths[frame_num,:,:] = depth_image.img
        depth_image.release()

    # Time images
    n_event_msgs = len(data.left_cam_readers['/davis/left/events'])
    msg_i = 0
    msg_i_last = 0
    e_i = 0
    delta = global_delta

    msg = data.left_cam_readers['/davis/left/events'][msg_i].message
    
    depth_ts_txt = open(base_name + '/text/depth_ts.txt', 'w')
    flow_ts_txt = open(base_name + '/text/flow_ts.txt', 'w')
    for frame_num in range(nframes):
        frame_time = timestamps[frame_num]
        ros_start_time = frame_time - delta
        part_left = delta

        msg_i = msg_i_last
        if (msg_i == n_event_msgs):
            break;

        cloud = []
        # Seek the target event msg
        while (msg_i < n_event_msgs):
            msg = data.left_cam_readers['/davis/left/events'][msg_i].message
            event_time = msg.header.stamp.to_sec()
            part_left = min(event_time, frame_time) - ros_start_time
            offset = max(0, event_time - frame_time)
            if (ros_start_time < event_time):
                break;
            msg_i += 1
        msg_i_last = msg_i

        e_start = msg.events[-1].ts.to_sec() - part_left
        e_i = 0

        cloud = [msg.events[0]]
        while ((cloud[-1].ts - cloud[0].ts).to_sec() < delta):
            for e_i in xrange(len(msg.events)):
                if ((cloud[-1].ts - cloud[0].ts).to_sec() >= delta):
                    break
                cloud.append(msg.events[e_i])

            if ((cloud[-1].ts - cloud[0].ts).to_sec() >= delta):
                break

            msg_i += 1
            msg = data.left_cam_readers['/davis/left/events'][msg_i].message
        current_ts = max(cloud[0].ts.to_sec(), cloud[-1].ts.to_sec()) - e0.ts.to_sec()

        timg, cimg, pimg = dvs_img(global_shape, cloud)
        timg = undistort_img(timg, flow.K, flow.distortion_coeffs)
        cimg = undistort_img(cimg, flow.K, flow.distortion_coeffs)
        pimg = undistort_img(pimg, flow.K, flow.distortion_coeffs)

        time_name = base_name + 'time_' + str(frame_num).rjust(10, '0') + ".png"
        count_name = base_name + 'cnt_' + str(frame_num).rjust(10, '0') + ".png"
        polar_name = base_name + 'polarity_' + str(frame_num).rjust(10, '0') + ".png"
        cmb_name = base_name + 'frame_' + str(frame_num).rjust(10, '0') + ".png"

        #cv2.imwrite(time_name, 255 * timg)
        #cv2.imwrite(count_name, cimg)
        #cv2.imwrite(polar_name, pimg)

        #cmb = np.dstack((cimg * global_scale_pp, timg * global_scale_t, pimg * global_scale_pn))
        #cv2.imwrite(cmb_name, cmb)


        #timg = normalize_img(timg)
        #cimg = normalize_img(cimg)
        #dpth = normalize_img(depths[frame_num])
        #ovrl = overlay(dpth, 50 * cimg)

        #cv2.imwrite(comb_name, 50 * np.hstack((timg, cimg, dpth)))
        #cv2.imwrite(comb_name, 50 * ovrl)

        #cv2.imwrite(comb_name, 50 * np.hstack((cimg, undistort_img(cimg, flow.K, flow.distortion_coeffs))))


        depth_name = base_name + '/eval/depth_' + str(frame_num).rjust(10, '0') + ".npy"
        depth_img = depths[frame_num]
        depth_img = depth_img[0:global_shape[0],0:global_shape[1]]

        np.save(depth_name, depth_img)

        depth_ts_txt.write('depth_' + str(frame_num).rjust(10, '0') + ".npy" + " " + str(current_ts) + '\n')


        # flow
        flow_x_dist, flow_y_dist = flow.compute_flow_single_frame(smoothed_Vs[frame_num],
                                                                  smoothed_Omegas[frame_num],
                                                                  depths[frame_num],
                                                                  dTs[frame_num])

        flow_name = base_name + '/eval/flow_' + str(frame_num).rjust(10, '0') + ".npy"

        flow_x_dist = flow_x_dist[0:global_shape[0],0:global_shape[1]]
        flow_y_dist = flow_y_dist[0:global_shape[0],0:global_shape[1]]
        np.save(flow_name, np.dstack((flow_x_dist, flow_y_dist)))

        flow_ts_txt.write('flow_' + str(frame_num).rjust(10, '0') + ".npy" + " " + str(current_ts) + '\n')

        print msg_i, frame_num, nframes, cloud[-1].ts.to_sec() - cloud[0].ts.to_sec()


    depth_ts_txt.close()
    flow_ts_txt.close()
    return x_flow_dist, y_flow_dist, timestamps, Vs, Omegas

def test_gt_flow():
    import calibration

    plt.close('all')

    cal = calibration.Calibration("indoor_flying")
    gtf = Flow(cal)
    
    p0 = np.array([0.,0.,0.])
    q0 = quaternion(1.0,0.0,0.0,0.0)

    depth = 10.*np.ones((cal.left_map.shape[0],cal.left_map.shape[1]))

    V, Omega = gtf.compute_velocity(p0,q0,p0,q0,0.1)
    x,y = gtf.compute_flow_single_frame(V, Omega, depth,0.1)

    fig = plt.figure()
    gtf.visualize_flow(x,y,fig)

    p1 = np.array([0.,0.25,0.5])
    q1 = quaternion(1.0,0.0,0.0,0.0)

    V, Omega = gtf.compute_velocity(p0,q0,p1,q1,0.1)
    print V, Omega
    x,y = gtf.compute_flow_single_frame(V, Omega, depth,0.1)

    fig = plt.figure()
    gtf.visualize_flow(x,y,fig)

    p1 = np.array([0.,-0.25,0.5])
    q1 = quaternion(1.0,0.0,0.0,0.0)

    V, Omega = gtf.compute_velocity(p0,q0,p1,q1,0.1)
    print V, Omega
    x,y = gtf.compute_flow_single_frame(V, Omega, depth,0.1)

    fig = plt.figure()
    gtf.visualize_flow(x,y,fig)
