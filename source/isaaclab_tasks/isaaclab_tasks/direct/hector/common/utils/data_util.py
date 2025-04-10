import torch



class HistoryBuffer:
    # Only push from back and pop from front
    def __init__(self, batch_size, capacity, feature_dim, dtype=torch.float, device='cpu'):
        """
        Initialize a batched deque.
        
        Args:
            batch_size (int): Number of deques to maintain (one per batch).
            capacity (int): Maximum number of elements per deque.
            feature_dim (int): Dimensionality of each deque element.
            dtype: Data type for the tensor.
            device: Device to store the data.
        """
        self.batch_size = batch_size
        self.capacity = capacity
        self.feature_dim = feature_dim
        
        # Create the underlying buffer: [batch_size, capacity, feature_dim]
        # self.buffer = torch.empty((batch_size, capacity, feature_dim), dtype=dtype, device=device)
        self.buffer = torch.zeros((batch_size, capacity, feature_dim), dtype=dtype, device=device)
        
        # Create head, tail, and size pointers for each batch; these are 1D tensors
        self.tail = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.size = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    def reset(self, env_id:torch.Tensor|None=None):
        """
        Reset the buffer to its initial state.
        """
        if env_id is None:
            env_id = torch.arange(self.batch_size)
        
        self.buffer[env_id, :, :] = torch.zeros((len(env_id), self.capacity, self.feature_dim), dtype=self.buffer.dtype, device=self.buffer.device)
        self.tail[env_id] = torch.zeros(len(env_id), dtype=self.tail.dtype, device=self.tail.device)
        self.size[env_id] = torch.zeros(len(env_id), dtype=self.size.dtype, device=self.size.device)
    
    def push(self, items:torch.Tensor, env_id:torch.Tensor|None=None):
        """
        Push a new element to the back of each deque.
        
        Args:
            items (Tensor): A tensor of shape [batch_size, feature_dim].
            
        Raises:
            IndexError: If any deque is full.
        """
        if env_id is None:
            env_id = torch.arange(self.batch_size)
        if items.shape[0] != env_id.shape[0] or items.shape[1] != self.feature_dim:
            raise ValueError("Items must have shape [batch_size, feature_dim].")
        if torch.any(self.size >= self.capacity):
            raise IndexError("At least one deque is full.")
        # Insert the new items at each batch's tail position.
        self.buffer[env_id, self.tail, :] = items
        # Update tail pointers: (tail + 1) mod capacity.
        self.tail[env_id] = (self.tail[env_id] + 1) % self.capacity
        # Increase the size of each deque.
        self.size[env_id] += 1
        
    def pop(self, env_id:torch.Tensor|None=None):
        """
        Pop an element from the front of each deque.
        
        Returns:
            Tensor: A tensor of shape [batch_size, feature_dim] containing the popped items.
            
        Raises:
            IndexError: If any deque is empty.
        """
        if env_id is None:
            env_id = torch.arange(self.batch_size)
            
        if torch.any(self.size[env_id] == 0):
            raise IndexError("Popping from empty que is not allowed.")
        items = self.buffer[env_id, 0, :].clone()
        
        self.buffer[env_id, 0, :] = torch.zeros_like(items)
        self.size[env_id] -= 1
        
        # shift tensor to left
        tensor1 = self.buffer[env_id, 0, :].clone()
        tensor2 = self.buffer[env_id, 1:, :].clone()
        self.buffer[env_id, :-1, :] = tensor2
        self.buffer[env_id, -1, :] = tensor1
        self.tail[env_id] = (self.tail[env_id]-1) % self.capacity
        return items
    
    def __getitem__(self, index):
        """
        Index into each deque (0 is the front).
        
        Args:
            index (int): The logical index in the deque.
            
        Returns:
            Tensor: A tensor of shape [batch_size, feature_dim] containing the items at the given index.
            
        Note:
            This implementation requires that index is valid for all deques. For simplicity, we check
            against the minimum current size across batches.
        """
        if not isinstance(index, int):
            raise TypeError("Index must be an integer.")
        if index < 0 or index >= torch.min(self.size).item():
            raise IndexError("Index out of range for at least one batch.")
        return self.buffer[:, index, :]
    
    @property
    def data_flat(self):
        """
        Returns:
            Tensor: The underlying buffer as a tensor of shape [batch_size, capacity, feature_dim].
        """
        return self.buffer.view(self.batch_size, -1)
    
    
if __name__ == "__main__":
    # Example usage
    batch_size = 1
    capacity = 5
    feature_dim = 2
    deques = HistoryBuffer(batch_size, capacity, feature_dim)
    # print(deques.size)  # [0, 0, 0]
    
    
    tensor1 = torch.ones(batch_size, feature_dim)
    tensor2 = 2 * torch.ones(batch_size, feature_dim)
    tensor3 = 3 * torch.ones(batch_size, feature_dim)
    tensor4 = 4 * torch.ones(batch_size, feature_dim)
    tensor5 = 5 * torch.ones(batch_size, feature_dim)
    
    
    deques.push(tensor1)
    deques.push(tensor2)
    deques.push(tensor3)
    deques.push(tensor4)
    deques.push(tensor5)
    print(deques.data_flat)
    
    deques.pop()
    print(deques.data_flat)
    
    tensor6 = 6 * torch.ones(batch_size, feature_dim)
    deques.push(tensor6)
    print(deques.data_flat)
    
    
    deques.pop()
    print(deques.data_flat)
    
    tensor7 = 7 * torch.ones(batch_size, feature_dim)
    deques.push(tensor7)
    print(deques.data_flat)
    
    
    # reset 
    print("reset")  
    deques.reset()
    print(deques.data_flat)
    
    tensor1 = torch.ones(batch_size, feature_dim)
    tensor2 = 2 * torch.ones(batch_size, feature_dim)
    tensor3 = 3 * torch.ones(batch_size, feature_dim)
    tensor4 = 4 * torch.ones(batch_size, feature_dim)
    tensor5 = 5 * torch.ones(batch_size, feature_dim)
    
    
    deques.push(tensor1)
    print(deques.data_flat)
    deques.push(tensor2)
    deques.push(tensor3)
    deques.push(tensor4)
    deques.push(tensor5)
    print(deques.data_flat)
    
    deques.pop()
    print(deques.data_flat)
    
    tensor6 = 6 * torch.ones(batch_size, feature_dim)
    deques.push(tensor6)
    print(deques.data_flat)
    
    
    deques.pop()
    print(deques.data_flat)
    
    tensor7 = 7 * torch.ones(batch_size, feature_dim)
    deques.push(tensor7)
    print(deques.data_flat)