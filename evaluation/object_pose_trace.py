"""Observe simulator object poses without issuing commands or changing scoring."""
import threading
import time


class ObjectPoseTrace:
    def __init__(self, names):
        import rospy
        from geometry_msgs.msg import PoseStamped
        self.names = list(names)
        self.lock = threading.Lock()
        self.latest = {}
        self.subscribers = [rospy.Subscriber('/mujoco/' + name + '/pose', PoseStamped,
            self.callback, callback_args=name, queue_size=1) for name in self.names]

    def callback(self, message, name):
        p = message.pose.position
        with self.lock:
            self.latest[name] = ([p.x, p.y, p.z], message.header.stamp.to_sec(), time.monotonic())

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            rows = [self.latest.get(name) for name in self.names]
        nan = float('nan')
        return {'position': [row[0] if row else [nan]*3 for row in rows],
                'source_time': [row[1] if row else nan for row in rows],
                'receive_age': [now-row[2] if row else nan for row in rows]}

    def close(self):
        for subscriber in self.subscribers:
            subscriber.unregister()
