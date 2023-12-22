import matplotlib.pyplot as plt
import torch


class MPC:
    def __init__(self, A, B, Q, R, P, N, x0, x_goal):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.P = P
        self.N = N
        self.x = x0
        self.x_goal = x_goal

    def predict(self, u) -> torch.Tensor:
        return self.A @ self.x + self.B @ u

    def cost(self, u):
        x_next = self.predict(u)
        return (
            # self.x.T @ self.Q @ (self.x - self.x_goal)
            (self.x - self.x_goal).T @ self.Q @ (self.x - self.x_goal)
            + u.T @ self.R @ u
            + (x_next - self.x_goal).T @ self.P @ (x_next - self.x_goal)
        ).sum()

    def control(self) -> torch.Tensor:
        u = torch.zeros((1, self.B.shape[0]), requires_grad=True)
        # u = torch.zeros((self.B.shape[0], 1), requires_grad=True)
        optimizer = torch.optim.Adam([u], lr=0.1)
        for _ in range(self.N):
            optimizer.zero_grad()
            cost = self.cost(u)
            cost.backward()
            optimizer.step()

        return u.detach()

    def apply_control(self, u) -> None:
        self.x = self.predict(u)


# Define system dynamics
A = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
B = torch.tensor([[1.0], [1.0]])

# Define cost matrices
Q = torch.eye(2)
R = torch.eye(1)
P = torch.eye(2)

# Define prediction horizon
N = 50

# Initial state
x0 = torch.tensor([[-10.0], [10.0]])

# Goal state
x_goal = torch.tensor([[25.0], [15.0]])

mpc = MPC(A, B, Q, R, P, N, x0, x_goal)

xs = []
us = []
iis = []
for i in range(100):
    # Get optimal control input
    u_opt = mpc.control()
    us.append(u_opt.T)
    print(f"{u_opt=}")

    mpc.apply_control(u_opt)
    # print(f"{mpc.x=}")
    xs.append(mpc.x)
    iis.append(i)


# plot x value against iteration
plt.plot(iis, [x[0] for x in xs], label="x")
plt.plot(iis, [u[0] for u in us], label="u_x")

# plot y value against iteration
plt.plot(iis, [x[1] for x in xs], label="y")
plt.plot(iis, [u[1] for u in us], label="u_y")
plt.legend()
plt.show()
