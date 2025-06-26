import torch
import torch.nn.functional as F


def gaussian_filter(image_tensor: torch.Tensor, kernel_size:int =5, sigma:float =1.0) -> torch.Tensor:
    """
    Applies a Gaussian filter to a PyTorch image tensor.

    Args:
        image_tensor (torch.Tensor): Input image tensor of shape (B, C, H, W).
        kernel_size (int): Size of the Gaussian kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        torch.Tensor: Filtered image tensor.
    """
    # Create 2D Gaussian kernel
    ax = torch.arange(kernel_size, dtype=torch.float32, device=image_tensor.device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    
    # Reshape to (1, 1, H, W) and repeat for each channel
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    num_channels = image_tensor.shape[1]
    kernel = kernel.repeat(num_channels, 1, 1, 1)

    # Apply depthwise convolution
    filtered_image = F.conv2d(image_tensor, kernel, padding=kernel_size // 2, groups=num_channels)
    return filtered_image

def laplacian_filter(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies a 3x3 Laplacian filter to a PyTorch image tensor.

    Args:
        image_tensor (torch.Tensor): Input image tensor of shape (B, C, H, W).

    Returns:
        torch.Tensor: Filtered image tensor.
    """
    # Define the Laplacian kernel
    laplacian_kernel = torch.tensor([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ], dtype=torch.float32, device=image_tensor.device).reshape(1, 1, 3, 3)

    num_channels = image_tensor.shape[1]
    laplacian_kernel = laplacian_kernel.repeat(num_channels, 1, 1, 1)
    filtered_image = F.conv2d(image_tensor, laplacian_kernel, padding='same', groups=num_channels)
    return filtered_image

def log_score_filter(image_tensor:torch.Tensor, alpha:float) -> torch.Tensor:
    """
    Applies gaussian filter and then Laplacian filter (Laplacian of Gaussian filter) to a PyTorch image tensor.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor of shape (B, C, H, W).
        alpha (float): How much to penalize the positive curveture.
    """
    # image = gaussian_filter(image_tensor, kernel_size=5, sigma=0.5)
    image = laplacian_filter(image_tensor)
    image = torch.minimum(torch.exp(-alpha * image), torch.tensor(1.0))
    return image