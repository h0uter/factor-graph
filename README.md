# factor-graph
library implementing barebones gbp for use with factor graphs. Based on work
- https://colab.research.google.com/drive/1-nrE95X4UC9FBLR0-cTnsIP_XhA_PZKW
- https://gaussianbp.github.io/


## Dev notes
### MVP 1
- only a goal layer.
- click to assign a new goal state $x^G$.
- agents can always communicate with each other.
- they optimize to push their goal state a minimum distance apart from the other agents' goal state.

![Alt text](doc/mvp1.png)

-> done see `examples/diverge_from_line.py`

### MVP 2
- now do it in 2d

### mvp 3 create a gui for it.
- [ ] make an event loop
- [ ]

### Further mvps:
- [ ] implement MPC.
- [ ] map landscape for mapping discrete actions.
- [ ] make plan for how to operate over a navigation graph instead of over a continuous space.
- [ ] go from goal layer to task layer.
- [ ]
