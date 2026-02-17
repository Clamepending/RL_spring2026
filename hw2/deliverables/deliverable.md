


## 3.2 Questions
– Which value estimator has better performance without advantage normalization: the trajectorycentric one, or the one using reward-to-go?
The reward to go looks slightly better, althought there looks like almost no difference in the small batch size experiment.

– Between the two value estimators, why do you think one is generally preferred over the other?
The reward to go seems to have less variance because it uses causality to remove the rewards before a decision. It also seems a tiny bit more elegant to implement programatically.

– Did advantage normalization help?
It seems to help quite a bit. the average reward gets to 200 quicker, and once there deviates less compared to non normalized runs.

– Did the batch size make an impact?
Yes, the smaller batch sizes seem to converge faster than the large experiments, but at the cost of having more instability after convergence.

command line configuration:
```bash
./experiment1.sh
```

## 4.2 questions

I changed the bgs to 2 from 5 for the experiment. 
a) The baseline learning curve is much more unstable (some spikes and a big uptrend towards the end) wiht only 2 steps. 
b) The eval avg return also decreased a lot at the beginning (around 1000) most likely due to the high baseline loss. The avg eval return went down when the baseline learning curve went up at the end suggesting the baseline got worse and made the policy worse as a result.

command line configuration:
```bash
./experiment1.sh
```