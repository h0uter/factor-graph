```julia
K = size(all_agents,1)
model = Model(Mosek.Optimizer)

@variable(model, u[1:vars.dim, 1:K, 1:vars.Thor])
@variable(model, p[1:vars.dim, 1:K, 1:vars.Thor+1])
# @expression(model, J4, )

@objective(model, Min, sum(1/(norm(all_agents[a].est_pos-all_agents[a].goal))* (p[:,a,t]-all_agents[a].goal)'I(vars.dim)(p[:,a,t]-all_agents[a].goal) for a in eachindex(all_agents[1:vars.k_c]), t in 2:vars.Thor+1))

@constraint(model, [a in eachindex(all_agents[1:vars.k_c]), t in 1:vars.Thor], p[:,a,t+1] == p[:,a,t] + u[:,a,t])
@constraint(model, [a in eachindex(all_agents[1:vars.k_c])], p[:,a,1] == all_agents[a].est_pos)
@constraint(model, [a in eachindex(all_agents[1:vars.k_c]), t in 1:vars.Thor], u[:,a,t]'*u[:,a,t]<=vars.umax^2)

# collision constraint
@constraint(model, [k in eachindex(all_agents[1:vars.k_c]), j in eachindex(all_agents), t in 2:vars.Thor+1; k!=j],
norm(all_agents[k].nom_traj[:,t]-all_agents[j].nom_traj[:,t])+
((all_agents[k].nom_traj[:,t]-all_agents[j].nom_traj[:,t])/norm(all_agents[k].nom_traj[:,t]-all_agents[j].nom_traj[:,t]))'*
((p[:,k,t]-p[:,j,t])-(all_agents[k].nom_traj[:,t]-all_agents[j].nom_traj[:,t]))-vars.r_coll>=0)

set_silent(model)


optimize!(model)
```
