# Object decoder decoding defaults

## Greedy EOF stopping

`MODEL.MASK_FORMER.OBJECT_DEC_EOF_THRESHOLD=0.5` applies strict EOF probability
>50% after vocabulary exclusions to inference and greedy RL references (RLOO
and best-of-N). Otherwise select the highest-probability mask. Zero restores
legacy individual-token argmax. Sampled RL rollouts retain categorical EOF
sampling and its original likelihood, so EOF keeps its policy-gradient signal.
