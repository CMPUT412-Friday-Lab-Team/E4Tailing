import math
import tf
import tf2_ros
import geometry_msgs.msg
from std_msgs.msg import Header
import numpy as np
import rospy
import os


HOST_NAME = os.environ["VEHICLE_NAME"]


TAG_DICT = {
    200: (.17, .17, math.pi * 1.25),
    201: (1.65, .17, math.pi * 1.75),
    94: (1.65, 2.84, math.pi * .25),
    93: (.17, 2.84, math.pi * .75),

    58: (.574, 1.259, math.pi * .5),
    162: (1.253, 1.253, math.pi * .0),
    153: (1.75, 1.252, math.pi * .5),
    133: (1.253, 1.755, math.pi * 1.5),
    169: (.574, 1.755, math.pi * 1.),
    62: (.075, 1.755, math.pi * 1.5),
}


def is_known_tag(tag_id):
    return tag_id in TAG_DICT


def extract_translation_and_quaternion_from_tag(tag_id):
    tag_x, tag_y, tag_angle = TAG_DICT[tag_id]
    q_world_to_tag = tf.transformations.quaternion_from_euler(tag_angle - .5 * math.pi, 0., -.5 * math.pi, 'rzyx')
    t_world_to_tag = np.array((tag_x, tag_y, .1), dtype=np.float32)

    return t_world_to_tag, q_world_to_tag


def broadcast_tag_tf(id, child_frame_id):
    print('broadcasting tag with id ', child_frame_id)
    translation, quaternion = extract_translation_and_quaternion_from_tag(id)

    br = tf2_ros.StaticTransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = 'world'
    t.child_frame_id = child_frame_id
    t.transform.translation = geometry_msgs.msg.Vector3(translation[0].item(), translation[1].item(), translation[2].item())
    t.transform.rotation = geometry_msgs.msg.Quaternion(quaternion[0].item(), quaternion[1].item(), quaternion[2].item(), quaternion[3].item())
    br.sendTransform(t)


def broadcast_tag_estimate(pose_t, pose_R, child_frame_id, publisher=None, isStatic=True):
    """
    broadcast a 'camera_april' (parent) -> 'estimated_tag' transformation
    """
    mat_R = np.identity(4)
    mat_R[0:3, 0:3] = pose_R
    q = tf.transformations.quaternion_from_matrix(mat_R)

    if publisher is None:
        if isStatic:
            br = tf2_ros.StaticTransformBroadcaster()
        else:
            br = tf2_ros.TransformBroadcaster()

    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = f'{HOST_NAME}/camera_april'
    t.child_frame_id = child_frame_id
    t.transform.translation = geometry_msgs.msg.Vector3(pose_t[0], pose_t[1], pose_t[2])
    t.transform.rotation = geometry_msgs.msg.Quaternion(q[0].item(), q[1].item(), q[2].item(), q[3].item())

    if publisher is None:
        br.sendTransform(t)
    else:
        publisher.publish(t)


def broadcast_robot_pos_from_estimate(pose_t, pose_R, tag_id, tfBuffer, publisher=None, isStatic=True):
    """
    broadcast robot position related to the tag
    """
    # 1.world to tag
    t_world_to_tag, q_world_to_tag = extract_translation_and_quaternion_from_tag(tag_id)
    R_world_to_tag = tf.transformations.quaternion_matrix(q_world_to_tag)

    # 2.tag to camera april
    mat_R = np.identity(4)
    mat_R[0:3, 0:3] = pose_R
    q_camera_april_to_tag = tf.transformations.quaternion_from_matrix(mat_R)
    q_tag_to_camera_april = tf.transformations.quaternion_inverse(q_camera_april_to_tag)
    t_camera_april_to_tag = np.array(pose_t, dtype=np.float32).reshape(3)
    R_tag_to_camera_april = tf.transformations.quaternion_matrix(q_tag_to_camera_april)

    # 3.camera april to footprint
    time = rospy.Time(0)  # see https://answers.ros.org/question/188023/tf-lookup-would-require-extrapolation-into-the-past/
    try:
        transform_camera_april_to_footprint = tfBuffer.lookup_transform(f'{HOST_NAME}/camera_april', f'{HOST_NAME}/footprint', time).transform
        t_camera_april_to_footprint = transform_camera_april_to_footprint.translation
        t_camera_april_to_footprint = np.array((t_camera_april_to_footprint.x, t_camera_april_to_footprint.y, t_camera_april_to_footprint.z), np.float32)
        q_camera_april_to_footprint = transform_camera_april_to_footprint.rotation
        q_camera_april_to_footprint = np.array((q_camera_april_to_footprint.x, q_camera_april_to_footprint.y, q_camera_april_to_footprint.z, q_camera_april_to_footprint.w), np.float32)
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException) as e:
        print('encountered exception when lookup:', e)
        t_camera_april_to_footprint = np.zeros(3)
        q_camera_april_to_footprint = np.array((0., 0., 0., 1.), dtype=np.float)

    # compose 1 and 2 (world to camera april)
    q_world_to_camera_april = tf.transformations.quaternion_multiply(q_world_to_tag, q_tag_to_camera_april)
    t_world_to_camera_april = t_world_to_tag + (R_world_to_tag[0:3, 0:3] @ (R_tag_to_camera_april[0:3, 0:3] @ -t_camera_april_to_tag.reshape((3, 1)))).reshape(3)
    R_world_to_camera_april = tf.transformations.quaternion_matrix(q_world_to_camera_april)

    # compose all (world to footprint)
    q = tf.transformations.quaternion_multiply(q_world_to_camera_april, q_camera_april_to_footprint)
    t = t_world_to_camera_april + (R_world_to_camera_april[0:3, 0:3] @ t_camera_april_to_footprint.reshape((3, 1))).reshape(3)

    # print('transformations:')
    # print('world_to_tag', t_world_to_tag, q_world_to_tag)
    # print('tag_to_camera_april', t_camera_april_to_tag, q_tag_to_camera_april)
    # print('camera_april_to_footprint', t_camera_april_to_footprint, q_camera_april_to_footprint)
    # print(R_world_to_tag)
    # print(R_world_to_camera_april)
    # print('world_to_camera_april', t_world_to_camera_april, q_world_to_camera_april)
    # print('overall', t, q)


    if publisher is None:
        if isStatic:
            br = tf2_ros.StaticTransformBroadcaster()
        else:
            br = tf2_ros.TransformBroadcaster()

    transformStamped = geometry_msgs.msg.TransformStamped()
    transformStamped.header.stamp = rospy.Time.now()
    transformStamped.header.frame_id = f'{HOST_NAME}/camera_april'
    transformStamped.child_frame_id = f'tag{id}'
    transformStamped.transform.translation = geometry_msgs.msg.Vector3(t[0].item(), t[1].item(), t[2].item())
    transformStamped.transform.rotation = geometry_msgs.msg.Quaternion(q[0].item(), q[1].item(), q[2].item(), q[3].item())

    if publisher is None:
        br.sendTransform(transformStamped)
    else:
        publisher.publish(transformStamped)


def broadcast_all_tags():
    for id in TAG_DICT.keys():
        broadcast_tag_tf(id, f'tag{id}')


