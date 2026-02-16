


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

