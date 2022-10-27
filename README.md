./run_mj examples.vitac_world.launch

1. Explore the world randomly.

2. Learn about positive and negative rewards from tactile sensing
        --> hypotheses: get stuck on always getting tactile feedback

3. Test with tactile curiosity
        --> the robot contacts surfaces and stuff and gets rewarded for the novelty, but gets bored after a while

4. The curiosity from tactile feedback drives the robot to contact objects on the world,
    which in turn drives the visual curiosity

        --> the robot achieves a good exploration

5. Show that the learnt dynamics model is useful for some manipulation task

        --> move the rope a to a specific configuration.
        --> reverse time.
                1. start with the robot in a given position
                2. get the robot exploring and moving the rope in random positions
                3. reverse the reward in time.

        --> start with robot in some random wierd positions e.g.
