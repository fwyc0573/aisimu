# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class Device():
    def __init__(self, name):
        # Name of device
        self.__name = name
        # The finish time of current node.
        self._next_finish_time = 0.0

    # Get the device name
    def name(self):
        return self.__name

    # Check whether the device is idle
    def is_idle(self):
        return True

    # Get the first completed node
    def get_next_node(self):
        return None

    def get_next_finish_time(self):
        return self._next_finish_time

    # Enqueue a new node into this device
    def enqueue_node(self, node, time_now):
        return

    # Dequeue the first completed node from the device.
    # Do not modify the attribute of the node, just modify info of device.
    def dequeue_node(self):
        return

    def __lt__(self, other):
        if self._next_finish_time != other._next_finish_time:
            return self._next_finish_time < other._next_finish_time
        else:
            return self.name() < other.name()
