


## Q 2.5

The eval returns very early on seem to be much lower than the train returns for mspacman. IDK why

## Q 3.5
– Does auto-tuning improve or achieve comparable performance to the fixed temperature?
Autotune achieves superior performance to fixed temperature.

– How does the temperature evolve during training? Does it increase, decrease, or stabilize?
The temperature decreases over time in a smooth looking asymptote close to 0.

– Why might the temperature change in this particular way for HalfCheetah?
The envirenment is probably not too sensitive to a high entropy policy, making the optimal policy able to support a big/decent entropy while keeping reward high and not costing much performance. This allows the entropy soft constraint to not "matter" that much for the optimal policy.

## Q 3.6
- Discuss how these results relate to overestimation bias.
The clipped Q value is significantly less than the unclipped Q value. The shapes of the eval returns and the q-values match: the clipped doesnt really platou while the unclipped platoud at around 200K steps. This suggests that the clipped Q values offered a more realistic assessment of the real q-values compared to the overestimated unclipped q values, and therefore provided a better gradient on which the policy was able to improve.