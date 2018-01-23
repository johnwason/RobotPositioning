#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
## END_SUB_TUTORIAL
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

import roslib

import rospkg
import numpy as np
from tf.transformations import *
from tf2_msgs.msg import TFMessage
from openravepy import *
import tf2_ros
import tf
import os



def rotx(theta):
  M = np.matrix([[1,0,0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
  return M

def roty(theta):
  M = np.matrix([[np.cos(theta), 0, np.sin(theta)], [0,1,0], [-np.sin(theta), 0, np.cos(theta)]])
  return M

def rotz(theta):
  M = np.matrix([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0,0,1]])
  return M


def FK_Matrix2(J):
  R01 = rotz(J[0])
  R12 = roty(J[1])
  R23 = roty(J[2])
  R34 = rotx(J[3])
  R45 = roty(J[4])
  R56 = rotx(J[5])
  R06 = np.dot(R01, np.dot(R12, np.dot(R23, np.dot(R34, np.dot(R45,R56)))))
  p12 = np.array([0.32,0,0.78]).reshape(3,1)
  p23 = np.array([0,0,1.075]).reshape(3,1)
  p34 = np.array([1.392,0,0.2]).reshape(3,1)
  p6T = np.array([0.15,0, -0.1]).reshape(3,1)
  p0T = np.dot(R01,p12) + np.dot(R01, np.dot(R12,p23)) + np.dot(R01, np.dot(R12,np.dot(R23,p34))) + np.dot(R06,p6T) + np.array([0.13,0, -0.40]).reshape(3,1)
  return np.dot(translation_matrix(p0T.flatten().tolist()[0]), np.r_[np.c_[R06,[0,0,0]],np.array([0,0,0,1]).reshape(1,4)].tolist())

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def transform2list(transform):
  values = [transform.translation.x,
            transform.translation.y,
            transform.translation.z,
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w]
  return values

class CollisionChecker:

  def __init__(self, gui=False):
    self.load_env()
    self.load_viewer(gui)
    self.load_distance_checker()
    rospy.loginfo('[CollisionChecker] Initialization finished.')

  def load_env(self):

    self.env = Environment()
    module = RaveCreateModule(self.env, 'urdf')

    self.bodies = {}
    self.joints = {}

    urdf_folder = rospkg.RosPack().get_path('abb_irb6640_support')+'/urdf/'

    names = ['irb6640_185_280_Testbed', 'Walls', 'box']
    urdfs = ['irb6640_185_280_or', 'Walls_or_NoWalls', 'box']
    model_urdf = {}
    for name, urdf in zip(names, urdfs):
      model_urdf[name] = urdf

    with self.env:
      for name in model_urdf:
        urdf = model_urdf[name]
        if urdf == 'box':
          body = RaveCreateKinBody(self.env, '')
          body.SetName(name)  
          body.InitFromBoxes(np.array([[0, 0, 0, 1.165, 0.1, 1.04]]), True) #0.6096, 0.01524, 0.3048
        else:
          body = self.env.GetKinBody(module.SendCommand('LoadURI '+urdf_folder+urdf+'.urdf'))
        self.env.AddKinBody(body)
        self.bodies[name] = body
        self.joints[name] = [ joint.GetName() for joint in body.GetJoints() ]




  def load_viewer(self, gui=False):
    if gui:
      self.env.SetViewer('qtcoin')

  def load_distance_checker(self):
    with self.env:
      options = CollisionOptions.Distance|CollisionOptions.Contacts
      if not self.env.GetCollisionChecker().SetCollisionOptions(options):
        rospy.loginfo('[CollisionChecker] Switching to pqp for collision distance information.')
        collisionChecker = RaveCreateCollisionChecker(self.env,'pqp')
        collisionChecker.SetCollisionOptions(options)
        self.env.SetCollisionChecker(collisionChecker)

  def check_safety(self, collision_poi, collision_env, joints=[]):

    # Find out background objects
    background = [ x for x in collision_env if x not in collision_poi ]

    # Update background transforms
    for bkg in background:
      if bkg in self.bodies:
        self.bodies[bkg].SetTransform(collision_env[bkg])

    # Update point of interest transforms
    for poi in collision_poi:
      if poi in self.bodies:
        self.bodies[poi].SetTransform(collision_poi[poi])
        #bodyexcluded.append(self.bodies[poi])

    # Update joint values
    for jnt in joints:
      if jnt in self.bodies:
        self.bodies[jnt].SetDOFValues(joints[jnt])

    # Check collisions
    report = CollisionReport()
    minDistance = np.infty
    with self.env:
      for poi in collision_poi:
        #for bkg in background + ['floor']:
        for bkg in background:
          if self.env.CheckCollision(link1=self.bodies[poi], link2=self.bodies[bkg], report=report):
            return False, report.minDistance
          elif minDistance > report.minDistance:
            minDistance = report.minDistance

    return True, minDistance

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def set_robot_pose(pose):
  broadcaster = tf2_ros.StaticTransformBroadcaster()
  static_transformStamped = geometry_msgs.msg.TransformStamped()
  
  static_transformStamped.header.stamp = rospy.Time.now()
  static_transformStamped.header.frame_id = "base"
  static_transformStamped.child_frame_id = "base_link"
  
  static_transformStamped.transform.translation.x = float(pose[0])
  static_transformStamped.transform.translation.y = float(pose[1])
  static_transformStamped.transform.translation.z = float(pose[2])
  
  static_transformStamped.transform.rotation.x = pose[3]
  static_transformStamped.transform.rotation.y = pose[4]
  static_transformStamped.transform.rotation.z = pose[5]
  static_transformStamped.transform.rotation.w = pose[6]
 
  for i in range(3):
    broadcaster.sendTransform(static_transformStamped)
    rospy.sleep(0.1)

def gen_pose_target(pose):
  pose_target = geometry_msgs.msg.Pose()
  pose_target.orientation.x = pose[3]
  pose_target.orientation.y = pose[4]
  pose_target.orientation.z = pose[5]
  pose_target.orientation.w = pose[6]
  pose_target.position.x = float(pose[0])
  pose_target.position.y = float(pose[1])
  pose_target.position.z = float(pose[2])
  return pose_target

def gen_pose_scene(pose):
  # Set the target pose in between the boxes and on the table
  target_pose = PoseStamped()
  target_pose.header.frame_id = "Wall/world"
  target_pose.pose.position.x = float(pose[0])
  target_pose.pose.position.y = float(pose[1])
  target_pose.pose.position.z = float(pose[2])
  target_pose.pose.orientation.x = pose[3]
  target_pose.pose.orientation.y = pose[4]
  target_pose.pose.orientation.z = pose[5]
  target_pose.pose.orientation.w = pose[6]
  return target_pose



def move_group_python_interface_tutorial():

  ## First initialize moveit_commander and rospy.
  print "============ Starting setup"
  moveit_commander.roscpp_initialize(sys.argv)
  rospy.init_node('collision_checker','move_group_python_interface_tutorial',
                  anonymous=True)
  
  ## Instantiate a RobotCommander object.  This object is an interface to
  ## the robot as a whole.
  robot = moveit_commander.RobotCommander()

  ## Instantiate a PlanningSceneInterface object.  This object is an interface
  ## to the world surrounding the robot.
  scene = moveit_commander.PlanningSceneInterface()

  ## Instantiate a MoveGroupCommander object.  This object is an interface
  ## to one group of joints.  In this case the group is the joints in the left
  ## arm.  This interface can be used to plan and execute motions on the left
  ## arm.
  group = moveit_commander.MoveGroupCommander("manipulator")

  #group.set_goal_position_tolerance(0.01)
  #group.set_goal_orientation_tolerance(0.1)
  group.allow_replanning(True)
  group.set_planner_id("ESTkConfigDefault") #RRTConnectkConfigDefault/SBLkConfigDefault/KPIECEkConfigDefault/BKPIECEkConfigDefault/LBKPIECEkConfigDefault/ ESTkConfigDefault
  #group.set_planning_time(15)
  group.set_num_planning_attempts(5)

  replanning_num = 5
  ## We create this DisplayTrajectory publisher which is used below to publish
  ## trajectories for RVIZ to visualize.
  display_trajectory_publisher = rospy.Publisher(
                                      '/move_group/display_planned_path',
                                      moveit_msgs.msg.DisplayTrajectory)

  ## Getting Basic Information
  ## ^^^^^^^^^^^^^^^^^^^^^^^^^
  ##
  ## We can get the name of the reference frame for this robot
  print "============ Reference frame: %s" % group.get_planning_frame()

  ## Sometimes for debugging it is useful to print the entire state of the
  ## robot.
  print "============ Printing robot state"
  print robot.get_current_state()
  print "============"

  print "============ Printing robot Pose"
  print group.get_current_pose().pose
  print "============"

  ## Set up the scene
  rospack = rospkg.RosPack()
  scene_path = rospack.get_path('abb_irb6640_support')
  wall_1 = os.path.join(scene_path,'meshes/Scene/T1.stl')
  wall_2 = os.path.join(scene_path,'meshes/Scene/T1.stl')
  wall_3 = os.path.join(scene_path,'meshes/Scene/T3.stl')
  Ceiling = os.path.join(scene_path,'meshes/Scene/TestBed_scene5.STL')
  
  # Set the target pose in between the boxes and on the table
  target_pose_panel = gen_pose_scene([1.862,0,1.325,0.5,-0.5,-0.5,0.5])#[0,-2,0.35,0.5,-0.5,-0.5,0.5]
  target_pose_panel.header.frame_id = 'base_link'
  #target_pose_wall_1 = gen_pose_scene([1.5384, -3.8262, 1,0,0,0,1])
  #target_pose_wall_2 = gen_pose_scene([1.5384, 2.6, 1,0,0,0,1])
  #target_pose_wall_3 = gen_pose_scene([-0.9, -0.6131, 1,0,0,0,1])
  #target_pose_Ceiling = gen_pose_scene([-0.9, -4.377, 0, 0,-0.707, -0.707,0])
  

  print "============ Starting Simulation ============"
  x_All = []
  y_All = []
  Dist_Mean_All = []
  Dist_Min_All = []
  cc = CollisionChecker(gui=False) 

  for i_x in np.linspace(-1,-0.72,8):  #-0.68,0.20,23
    for i_y in np.linspace(-1.2,1.2,51): #(-1,1,51)

      scene.remove_world_object('target')
      #scene.remove_world_object('Wall_1')
      #scene.remove_world_object('Wall_2')
      #scene.remove_world_object('Wall_3')
      #scene.remove_world_object('Ceiling')
      rospy.sleep(2)
      print "============ Iteration ============"
      print i_x, i_y
      np.savetxt('ScoreMap.out',(x_All,y_All,Dist_Mean_All,Dist_Min_All))

      Dist = []
      robot_pose = [i_x, i_y, 0, 0, 0, 0, 1]
      set_robot_pose(robot_pose)
      # Add the target object to the scene
      scene.add_box('target', target_pose_panel, [2.33, 0.2, 2.08] )
      #scene.add_mesh('Wall_1', target_pose_wall_1, wall_1 , size =(0.5,0.5,0.5))
      #scene.add_mesh('Wall_2', target_pose_wall_2, wall_2 , size =(0.5,0.5,0.5))
      #scene.add_mesh('Wall_3', target_pose_wall_3, wall_3 , size =(0.5,0.5,0.5))
      #scene.add_mesh('Ceiling', target_pose_Ceiling, Ceiling)
      ## Wait for RVIZ to initialize. This sleep is ONLY to allow Rviz to come up.
      print "============ Waiting for RVIZ..."
      rospy.sleep(2)

      print "============ Generating plan REST ============"
      group.clear_pose_targets()
      pose_target = gen_pose_target([1.862,0,1.955,0,-1,0,0])
      group.set_pose_target(pose_target)
      planR = group.plan()


      cnt = 0;
      while( (not planR.joint_trajectory.points) & (cnt< replanning_num)):
        cnt = cnt+1;
        print "============ Generating plan again"
        planR = group.plan()

      if (cnt == 5):
        Dist=-1
        group.detach_object('link_6')
        x_All.append(i_x)
        y_All.append(i_y)
        Dist_Mean_All.append(np.mean(Dist))
        Dist_Min_All.append(np.min(Dist))   
        continue

      rospy.sleep(1)
      print "============ Executing plan REST"
      group.execute(planR)
      print "============ Waiting while plan REST is Executed..."
      rospy.sleep(1)



      T1 = np.dot(translation_matrix(robot_pose[0:3]), quaternion_matrix(robot_pose[3:7]))
      T2 = np.dot(translation_matrix([0,0,0]), quaternion_matrix([0,0,0,1]))
      T3 = np.dot(T1,np.dot(translation_matrix([0,-2,0.35]), quaternion_matrix([0.5, 0.5, 0.5, 0.5])))

      group.attach_object('target','link_6')
      rospy.sleep(2)

      collision_env = { 'Walls'   : T2 }



      print "============ Generating plan 2 ============"
      group.clear_pose_targets()
      pose_target = gen_pose_target([1.7-robot_pose[0],-1.165-robot_pose[1],0.62,0,-1,0,0])
      group.set_pose_target(pose_target)

      ## Now, we call the planner to compute the plan
      ## and visualize it if successful
      ## Note that we are just planning, not asking move_group 
      ## to actually move the robot
      plan2 = group.plan()

      cnt = 0;
      while( (not plan2.joint_trajectory.points) & (cnt< replanning_num)):
        cnt = cnt+1;
        print "============ Generating plan again"
        plan2 = group.plan()

      if (cnt == 5):
        Dist=-1
        group.detach_object('link_6')
        x_All.append(i_x)
        y_All.append(i_y)
        Dist_Mean_All.append(np.mean(Dist))
        Dist_Min_All.append(np.min(Dist))   
        continue
      #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      for i_plan in range(0,len(plan2.joint_trajectory.points)):
        tmp_array = plan2.joint_trajectory.points[i_plan].positions
        T3 = np.dot(T1,np.dot(FK_Matrix2(tmp_array), quaternion_matrix([0.707, 0.707, 0, 0])))
        #T3 = FK_Matrix(tmp_array)
        joints = { 'irb6640_185_280_Testbed' : tmp_array }

        collision_poi = { 'irb6640_185_280_Testbed' : T1, 'box' : T3 }

        tmp_result = cc.check_safety(collision_poi, collision_env, joints)
        print 'Safe (0.001 tolerance) =',tmp_result
        Dist.append(tmp_result[1])
      #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      rospy.sleep(1)
      print "============ Executing plan2"
      group.execute(plan2)
      print "============ Waiting while plan2 is Executed..."
      rospy.sleep(1)


      print "============ Generating plan REST ============"
      group.clear_pose_targets()
      pose_target = gen_pose_target([1.862,0,1.955,0,-1,0,0])
      group.set_pose_target(pose_target)
      planR = group.plan()

      cnt = 0;
      while( (not planR.joint_trajectory.points) & (cnt< replanning_num)):
        cnt = cnt+1;
        print "============ Generating plan again"
        planR = group.plan()
      
      if (cnt == 5):
        Dist=-1
        group.detach_object('link_6')
        x_All.append(i_x)
        y_All.append(i_y)
        Dist_Mean_All.append(np.mean(Dist))
        Dist_Min_All.append(np.min(Dist))   
        continue
      #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      for i_plan in range(0,len(planR.joint_trajectory.points)):
        tmp_array = planR.joint_trajectory.points[i_plan].positions
        T3 = np.dot(T1,np.dot(FK_Matrix2(tmp_array), quaternion_matrix([0.707, 0.707, 0, 0])))
        joints = { 'irb6640_185_280_Testbed' : tmp_array }

        collision_poi = { 'irb6640_185_280_Testbed' : T1, 'box' : T3 }

        tmp_result = cc.check_safety(collision_poi, collision_env, joints)
        print 'Safe (0.001 tolerance) =',tmp_result
        Dist.append(tmp_result[1])
      #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      rospy.sleep(1)
      print "============ Executing plan REST"
      group.execute(planR)
      print "============ Waiting while plan REST is Executed..."
      rospy.sleep(1)

      print "============ Generating plan 3 ============"
      group.clear_pose_targets()
      pose_target = gen_pose_target([1.7-robot_pose[0],1.165-robot_pose[1],0.62,0,-1,0,0])
      group.set_pose_target(pose_target)
      plan3 = group.plan()

      cnt = 0;
      while( (not plan3.joint_trajectory.points) & (cnt< replanning_num)):
        cnt = cnt+1;
        print "============ Generating plan again"
        plan3 = group.plan()

      if (cnt == 5):
        Dist=-1
        group.detach_object('link_6')
        x_All.append(i_x)
        y_All.append(i_y)
        Dist_Mean_All.append(np.mean(Dist))
        Dist_Min_All.append(np.min(Dist))   
        continue
      #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      for i_plan in range(0,len(plan3.joint_trajectory.points)):
        tmp_array = plan3.joint_trajectory.points[i_plan].positions
        T3 = np.dot(T1,np.dot(FK_Matrix2(tmp_array), quaternion_matrix([0.707, 0.707, 0, 0])))
        joints = { 'irb6640_185_280_Testbed' : tmp_array }

        collision_poi = { 'irb6640_185_280_Testbed' : T1, 'box' : T3 }

        tmp_result = cc.check_safety(collision_poi, collision_env, joints)
        print 'Safe (0.001 tolerance) =',tmp_result
        Dist.append(tmp_result[1])
      #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      rospy.sleep(1)
      print "============ Executing plan3"
      group.execute(plan3)
      print "============ Waiting while plan3 is Executed..."
      rospy.sleep(1)


      print "============ Generating plan REST ============"
      group.clear_pose_targets()
      pose_target = gen_pose_target([1.862,0,1.955,0,-1,0,0])
      group.set_pose_target(pose_target)
      planR = group.plan()

      cnt = 0;
      while( (not planR.joint_trajectory.points) & (cnt< replanning_num)):
        cnt = cnt+1;
        print "============ Generating plan again"
        planR = group.plan()

      if (cnt == 5):
        Dist=-1
        group.detach_object('link_6')
        x_All.append(i_x)
        y_All.append(i_y)
        Dist_Mean_All.append(np.mean(Dist))
        Dist_Min_All.append(np.min(Dist))   
        continue
      #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      for i_plan in range(0,len(planR.joint_trajectory.points)):
        tmp_array = planR.joint_trajectory.points[i_plan].positions
        T3 = np.dot(T1,np.dot(FK_Matrix2(tmp_array), quaternion_matrix([0.707, 0.707, 0, 0])))
        joints = { 'irb6640_185_280_Testbed' : tmp_array }
        collision_poi = { 'irb6640_185_280_Testbed' : T1, 'box' : T3 }

        tmp_result = cc.check_safety(collision_poi, collision_env, joints)
        print 'Safe (0.001 tolerance) =',tmp_result
        Dist.append(tmp_result[1])
      #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      rospy.sleep(1)
      print "============ Executing plan REST"
      group.execute(planR)
      print "============ Waiting while plan REST is Executed..."
      rospy.sleep(1)



      ## Adding/Removing Objects and Attaching/Detaching Objects
      ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      ## First, we will define the collision object message
      group.detach_object('link_6')
      x_All.append(i_x)
      y_All.append(i_y)
      Dist_Mean_All.append(np.mean(Dist))
      Dist_Min_All.append(np.min(Dist))   
      ## END


  scene.remove_world_object('target')
  #scene.remove_world_object('Wall_1')
  #scene.remove_world_object('Wall_2')
  #scene.remove_world_object('Wall_3')
  #scene.remove_world_object('Ceiling')
  rospy.sleep(2)



  print "============ STOPPING"
  collision_object = moveit_msgs.msg.CollisionObject()

  np.savetxt('ScoreMap.out',(x_All,y_All,Dist_Mean_All,Dist_Min_All))

  ## When finished shut down moveit_commander.
  moveit_commander.roscpp_shutdown()


if __name__=='__main__':
  
  try:

    move_group_python_interface_tutorial()
  except rospy.ROSInterruptException:
    pass

