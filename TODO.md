-[x] clean up(Rogier & Frank):
    -[x] remove old environments not used anymore
    -[x] remove minigridworld stuff
-[x] implement function to calculate true Q values & plotting functionality for visualizing max bias (Pieter & Yun)

-[ ] for episode return and episode lengths fix xlabel from steps to episode(Pieter)
    -find a way to automatically (or manually) clip x-axis to where slowest algorithm converges to optimal performance (Pieter, make separate py file for reading in pickle)
-[ ] make config option that allows for not saving Q table data (Pieter)
-[ ] save average return per step plot (in main, Pieter)
-[ ] titles of plots (Pieter):
    -“learning curves for ENV, HYPERPARAMS”
    -“V values found by ALG in ENV at final step”
    -“Max Q value actions  found by ALG in ENV at final step”

-[ ] store all information for both vanilla and double (Pieter)
    -plots (and saves)
-[ ] policy plots if multiple argmaxes, plot all corresponding arrows(Pieter)

-[ ] make hyperparameters algorithm specific(Pieter)


Discussion needed:
-[ ] think about what error bars represent (plot standard deviation?)
-[ ] hyperparameter tuning code:
    -how are we going to optimize alpha?
        -paper: the learning rate was either 
            -linear(alpha depends on state and action, 1/n at time=t of (s,a)) or 
            -polynomial()
    -no gamma optimization
    -epsilon is useful to optimize as well (feasible? different for Vanilla and Double?)
        -(epsilon at time=0,Beta) thing (Rogier)
    -suggestion: use number of steps to delta within optimality (+-0 episode return) as performance metric for hyperparameter optimization

-[ ] Q-learning is off-policy!! although in specific case of Q-learning there is no explicit target and behavior policy, 
this distinction is still relevant!
    -learning curves depend on epsilon of the policy, do we want to look at determistic policy performance or epsilon-greedy policy
    -true Q value calculation: use deterministic policy or epsilon greedy one
        -when fair comparison to current estimates?
            -both estimates and true values same policy? Not possible, estimates always from epsilon greedy policy




