
import torch
from torch.optim import Adam


# MOMENT MATCHER ###############################################################


class MomentMatcher:

    def __init__(self,
                 K_samples,
                 r_samples):

        # parameters
        self.NUM_FITS = 10
        self.MAX_STEPS = 10000
        self.REGULARIZATION_LOSS_CONTRIBUTION = 0.5
        # TODO: how many moments do we need to use? (i just used num_parameters)
        self.CONVERGENCE_CHECK_EVERY = 100
        self.CONVERGENCE_TOL = 0.01

        # keep track of provided samples
        self.K_samples = torch.tensor(K_samples, dtype=torch.float64)
        self.r_samples = torch.tensor(r_samples, dtype=torch.float64)

        # determine number of parameters to estimate
        self.num_parameters = r_samples.shape[1]

        # compute expected values of samples
        self.E_K = self.K_samples.mean(0)
        self.E_r = self.r_samples.mean(0)

        # compute differences to expected value of K_samples
        self.diff_K = self.K_samples - self.E_K

        # create parameters to estimate
        self.K = torch.empty(self.num_parameters, dtype=torch.float64)
        self.K = torch.nn.Parameter(self.K)

    def get_moment_matching_loss(self):

        # compute expected value of r*K
        E_rK = torch.dot(self.E_r, self.K)

        # determine loss for first moment
        loss = (self.E_K - E_rK).pow(2)

        # determine loss for higher-order moments
        # (range will be empty if self.num_parameters = 1)
        for m in range(2, self.num_parameters+1):

            # compute LHS m-th centralized moment
            LHS = self.diff_K.pow(m).mean(0)

            # compute RHS m-th centralized moment
            RHS = (torch.matmul(self.r_samples, self.K) - E_rK).pow(m).mean(0)

            # accumulate squared difference of centralized moments
            loss += (LHS-RHS).pow(2)

        return loss

    def get_regularization_loss(self):

        # create regularization loss tensor
        loss = self.K.pow(2).sum()

        return loss

    def get_loss(self):

        # get loss contributions
        moment_matching_loss = self.get_moment_matching_loss()
        regularization_loss = self.get_regularization_loss()

        # make regularization loss to be 50% of moment_matching_loss
        L_mm = moment_matching_loss.item()
        L_r = regularization_loss.item()
        reg_factor = (1.0/L_r)*L_mm*self.REGULARIZATION_LOSS_CONTRIBUTION

        # compute weighted combination of losses
        loss = self.get_moment_matching_loss() + \
               reg_factor*self.get_regularization_loss()

        return loss

    def train(self,init=None):

        # initialize parameters
        if init is None:
            torch.nn.init.normal_(self.K, 0.0, 1.0)
        else:
            self.K.data = init

        # initialize optimizer
        optimizer = Adam([self.K])

        # keep track of perevious parameter configuration
        prev_K = self.K.clone()

        # train for max_steps iterations
        for i in range(1, self.MAX_STEPS+1):

            optimizer.zero_grad()   # zero-out the parameter gradients
            loss = self.get_loss()  # compute loss
            loss.backward()         # compute gradients through backprop
            optimizer.step()        # do gradient step

            # do solution convergence check every 100 STEPS
            if i % self.CONVERGENCE_CHECK_EVERY == 0:
                convergence_cond = torch.allclose(prev_K,
                                                  self.K,
                                                  atol=self.CONVERGENCE_TOL,
                                                  rtol=self.CONVERGENCE_TOL)
                if convergence_cond:
                    # if current solution hasn't changed much, abort training
                    break
                else:
                    # otherwise, continue training
                    prev_K = self.K.clone()

        # copy solution
        solution = self.K.clone()

        print('Fitted curvatures in ' + \
              str(i)+'/'+str(self.MAX_STEPS)+' steps, ' + \
              '\tLoss: '+str(loss.item()) + ', ' + \
              '\tSolution: ' + str(solution.data))

        return solution

    def fit(self):

        # keep track of found solutions
        solutions = []

        # train the model NUM_FITS times
        for i in range(self.NUM_FITS):
            # train to get one solution for self.K
            solution = self.train()
            # append solution to list of solutions
            solutions.append(solution)

        # concatenate solutions
        solutions = torch.cat(solutions).view(-1, self.num_parameters)

        # compute average of solutions
        avg_solution = solutions.mean(0)

        # train again with average as initialization
        final_solution = self.train(init=avg_solution)

        # convert to numpy array
        final_solution = final_solution.data.numpy()

        return final_solution

