# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

"""Package containing task implementations for various robotic environments."""

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##


# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
