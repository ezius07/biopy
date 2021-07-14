import torch

available_distr = {
    
    'flower_2d_synthetic': {    
        'means': torch.tensor(
            [
                [-5., 0.],
                [0., 5.],
                [5., 0.],
                [-2.5, -6.],
                [2.5, -6.]
            ]
        ),
        'covariances': torch.tensor(
            [
                [[4., 0.],
                 [0., .2]],
                [[.2, 0.],
                 [0., 4.]],
                [[4., 0.],
                 [0., .2]],
                [[.5, 1.3],
                 [1.3, 4.]],
                [[.5, -1.3], 
                 [-1.3, 4.]]
            ]
        )
    },
    
    'circles_2d_synthetic': {
        'means' : torch.tensor(
            [
                [-5., 5.],
                [5., 5.],
                [0., 0.],
                [-5., -5.],
                [5., -5.]
            ]
        ),
        'covariances': torch.tensor(
            [
                [[2., 0.],
                 [0., 2.]],
                [[2., 0.],
                 [0., 2.]],
                [[2., 0.],
                 [0., 2.]],
                [[2., 0.],
                 [0., 2.]],
                [[2., 0.],
                 [0., 2.]]
            ]
        )
    },
    
}