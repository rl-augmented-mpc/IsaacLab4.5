import torch

def compute_nominal_stats(nominal_data: torch.Tensor):
    """
    Compute the mean and covariance matrix from nominal data.
    
    Args:
        nominal_data (torch.Tensor): A tensor of shape (N, D) where N is the number 
                                     of samples and D is the number of features (e.g., 
                                     [v_slip, h_com_deviation, h_foot_deviation]).
                                     
    Returns:
        mu (torch.Tensor): Mean vector of shape (D,).
        cov (torch.Tensor): Covariance matrix of shape (D, D).
    """
    mu = torch.mean(nominal_data, dim=0, keepdim=True)
    centered = nominal_data - mu
    cov = centered.T @ centered / (nominal_data.shape[0] - 1)
    cov = cov.unsqueeze(0)
    
    nll_mean = (centered.unsqueeze(2).transpose(1, 2) @ torch.linalg.inv(cov) @ centered.unsqueeze(2)).squeeze(1, 2)
    nll_mean = nll_mean.mean().unsqueeze(0)
    return mu, cov, nll_mean

def ood_distance(x_new: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor)->torch.Tensor:
    """
    Compute the Mahalanobis distance for a new observation.
    
    Args:
        x_new (torch.Tensor): New observation tensor of shape (D,).
        mu (torch.Tensor): Mean vector of the nominal data (D,).
        cov (torch.Tensor): Covariance matrix of the nominal data (D, D).
    
    Returns:
        score (torch.Tensor): The Mahalanobis distance (a scalar tensor).
    """
    # mahalanobis distance 
    # d = sqrt((x - mu)^T * cov^-1 * (x - mu))
    # or negative log likelihood 
    # d = 0.5 * (x - mu)^T * cov^-1 * (x - mu) + 0.5 * log(det(cov))
    diff = x_new - mu
    diff = diff * (diff > 0) # only consider positive deviations
    cov_inv = torch.linalg.inv(cov+1e-6*torch.eye(cov.shape[0], device=cov.device).unsqueeze(0)) # regularize covariance matrix to make positive definite
    
    d = torch.sqrt((diff.unsqueeze(2).transpose(1, 2) @ cov_inv @ diff.unsqueeze(2)).squeeze(1, 2))
    # d = (0.5 * diff.unsqueeze(2).transpose(1, 2) @ cov_inv @ diff.unsqueeze(2)).squeeze(1, 2)
    return d

def compute_regularization_lambda(anomaly_score: torch.Tensor,
                                  lam_max: float = 1.0, 
                                  alpha: float = 0.1):
    """
    Map the anomaly score to a PPO regularization coefficient lambda.
    
    Args:
        anomaly_score (torch.Tensor): The computed anomaly score (Mahalanobis distance).
        lam_base (float): The baseline (minimal) regularization coefficient.
        alpha (float): Scaling factor for linear mapping.
        lam_max (float): Maximum regularization coefficient (for nonlinear mapping).
        beta (float): Scaling factor for the sigmoid mapping.
        tau (float): Threshold for the sigmoid mapping.
        mapping (str): Either 'linear' or 'sigmoid' to select the mapping type.
        
    Returns:
        lam (torch.Tensor): The resulting regularization coefficient.
    """
    # lam = lam_max * torch.exp(-0.5*torch.square(alpha*anomaly_score))
    ratio = torch.clamp(1/(anomaly_score+1e-6), 0, 1)
    lam = lam_max*ratio # linear mapping
    # lam = lam_max * torch.exp(-0.1*anomaly_score)
    return lam

if __name__ == "__main__":
    import numpy as np
    import pickle
    
    with open("./anomaly.pkl", "rb") as f:
        data = pickle.load(f)
    
    data = np.array(data)
    data = torch.from_numpy(data)
    data = data[:, :, 1:-1] # cut off the first and last columns
    data = data.reshape(-1, data.shape[-1])
    print(data.shape)
    
    # data_test = np.array(data_test)
    # data_test = torch.from_numpy(data_test)
    # data_test = data_test[:, :, 1:-1] # cut off the first and last columns
    # data_test = data_test.reshape(-1, data_test.shape[-1])
    
    # Assume nominal_data is a tensor with each row corresponding to:
    # [(h_com_deviation, h_foot_deviation, v_slip)]
    mu, cov, nll_mean = compute_nominal_stats(data)
    print(mu)
    print(cov)
    print(nll_mean)
    
    # anomaly = nll_score(data_test, mu, cov, nll_mean)
    # lam_reg = compute_regularization_lambda(anomaly, lam_max=0.1, alpha=0.1)
    # print(lam_reg[:100])
    # print(f"Regularization Coefficient (Linear Mapping): {lam_linear.item():.4f}")
    
    
    
    # import matplotlib.pyplot as plt
    
    # # height diff
    # x = np.linspace(0, 1, 100)
    # y = np.exp(0.5*x)-1
    # y2 = np.log(1+x)
    
    # plt.plot(x, x, label="Linear")
    # plt.plot(x, y, label="Exponential")
    # plt.plot(x, y2, label="Logarithmic")
    # plt.legend()
    # plt.show()