#!/bin/bash

killall -9 gzserver
killall -9 gzclient
killall -9 roscore
killall -9 rosmaster
rosnode kill -a
