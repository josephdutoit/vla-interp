Now that I have a sparse autoencoder trained, I want to see if I can use it to steer features. 
I have found the following features that activate on the object X for the task "pick up the X and put it in the basket". I want to see if by suppressing the other features and boosting the features on some object, I can steer the model to pick up that object instead of the other one.

The feautures I have found and the corresponding objects are:
45366 tomato sauce
8447 cream cheese
2986 ketchup 
26272 salad dressing
40654 milk

To do this we will need to load the libero simulator, run it through the model up to layer 11, then run it through the SAE while boosting the feature and suppressing the others. Then we will run it through the rest of the model and see if it picks up the correct object.

You may want to look at /home/jcdutoit/CS601/vla-interp/vla0-trl/scripts/eval.py to see how to get load the simulator. We will need to get the entire episode for that frame. It may also be helpful to look at /home/jcdutoit/CS601/vla-interp/vla0-trl/scripts/identify_features.py to see how to run the SAE. 

The output should be a video of the model in the simulator for each of the 5 features, showing that it picks up the correct object.

Let me know if you have any questions or if you want to discuss the plan further.

