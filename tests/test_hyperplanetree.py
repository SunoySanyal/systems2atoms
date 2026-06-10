import torch
torch_device = 'cpu'
from systems2atoms.hyperplanetree import *

def generate_function(x, y, noise_scale = 0.1):
    a = x/1.3 + y   
    f = a*x*y + a**2 + 4*torch.cos(3*a)
    
    f += torch.normal(mean = torch.zeros_like(x), std = noise_scale)

    return f.to(torch_device)

def generate_sparse_linear_data(n_samples=80, n_features=6):
    generator = torch.Generator()
    generator.manual_seed(0)
    X = torch.randn(n_samples, n_features, generator=generator)
    noise = 0.01 * torch.randn(n_samples, generator=generator)
    y = 3.0 * X[:, 0] - 2.0 * X[:, 2] + noise
    return X.type(torch.float), y.type(torch.float)

def leaf_coefficients(model):
    coefs = []
    for leaf in model._leaves.values():
        coef = torch.as_tensor(leaf.model.coef_, dtype=torch.float)
        coefs.append(coef.reshape(-1))
    return coefs

def test_hyperplanetree():
    # Generate sampling points
    x0 = torch.linspace(-3, 3, 20, device = torch_device)
    x1 = torch.linspace(-3, 3, 20, device = torch_device)
    X0, X1 = torch.meshgrid(x0, x1, indexing='ij')

    # Generate features and labels tensors
    features = torch.vstack((X0.flatten(), X1.flatten())).T.type(torch.float)
    y = generate_function(X0, X1).type(torch.float).flatten()

    shuffle = torch.randperm(len(y))

    train_indices = shuffle[:int(0.8*len(shuffle))]
    test_indices = shuffle[int(0.8*len(shuffle)):]

    train_features = features[train_indices]
    test_features = features[test_indices]
    train_y = y[train_indices]
    test_y = y[test_indices]

    model = LinearTreeRegressor()
    model.fit(train_features, train_y)
    y_pred = model.predict(test_features.to(torch_device))
    leaves = len(model)
    assert leaves > 2
    assert max(y_pred) - min(y_pred) > 0

    model = HyperplaneTreeRegressor()
    model.fit(train_features, train_y)
    y_pred = model.predict(test_features.to(torch_device))
    leaves = len(model)
    assert leaves > 2
    assert max(y_pred) - min(y_pred) > 0

def test_formulations():
    model = HyperplaneTreeRegressor()
    features = torch.randn(10, 2)
    labels = torch.randn(10)
    model.fit(features, labels)
    
    definition = HyperplaneTreeDefinition(
        model,
        input_bounds_matrix = torch.stack([
            torch.min(features, dim=0).values,
            torch.max(features, dim=0).values,
        ]).T,
    )

    formulation = HyperplaneTreeGDPFormulation(definition)
    formulation = HyperplaneTreeHybridBigMFormulation(definition)

def test_leaf_regularization_modes_fit_and_report_coefficients():
    features, labels = generate_sparse_linear_data()

    for regularization in ["ridge", "lasso", "elasticnet"]:
        model = LinearTreeRegressor(
            max_depth=1,
            min_samples_leaf=20,
            max_bins=4,
            disable_tqdm=True,
            leaf_regularization=regularization,
            leaf_alpha=0.05,
            leaf_l1_ratio=0.5,
            max_iter=10000,
            tol=1e-6,
            random_state=0,
        )
        model.fit(features, labels)

        predictions = model.predict(features[:5])
        assert predictions.reshape(-1).shape == labels[:5].shape
        assert len(model) >= 1

        for coef in leaf_coefficients(model):
            assert coef.numel() == features.shape[1]

def test_hyperplanetree_with_elasticnet_leaves():
    features, labels = generate_sparse_linear_data()
    model = HyperplaneTreeRegressor(
        max_depth=2,
        min_samples_leaf=15,
        max_bins=4,
        disable_tqdm=True,
        leaf_regularization="elasticnet",
        leaf_alpha=0.01,
        leaf_l1_ratio=0.5,
        max_iter=10000,
        tol=1e-6,
        random_state=0,
    )
    model.fit(features, labels)

    predictions = model.predict(features[:5])
    assert predictions.reshape(-1).shape == labels[:5].shape
    assert len(model) >= 1

def test_sparse_leaf_uncertainty_raises_clear_error():
    import pytest

    features, labels = generate_sparse_linear_data()
    model = LinearTreeRegressor(
        max_depth=1,
        min_samples_leaf=20,
        max_bins=4,
        disable_tqdm=True,
        leaf_regularization="elasticnet",
        leaf_alpha=0.05,
        random_state=0,
    )
    model.fit(features, labels)

    with pytest.raises(NotImplementedError, match="ridge"):
        model.uncertainty(features[:5])

def test_l1_leaf_regularization_can_zero_coefficients():
    features, labels = generate_sparse_linear_data()
    model = LinearTreeRegressor(
        max_depth=1,
        min_samples_leaf=20,
        max_bins=4,
        disable_tqdm=True,
        leaf_regularization="lasso",
        leaf_alpha=0.5,
        max_iter=10000,
        tol=1e-6,
        random_state=0,
    )

    model.fit(features, labels)
    coefficients = torch.cat(leaf_coefficients(model))

    assert torch.any(torch.abs(coefficients) <= 1e-6)
