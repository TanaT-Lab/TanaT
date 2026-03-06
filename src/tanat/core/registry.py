#!/usr/bin/env python3
"""
Factory registry.
"""

# Registry for container factories
# Dict { container_type: factory_func }
_POOL_FACTORIES = {}


def register_factory(container_type, factory_func):
    """Registers a factory function for a specific container type."""
    _POOL_FACTORIES[container_type] = factory_func


def build_pool(container_type, store_instance):
    """Retrieves the factory function and creates the Pool."""
    factory_fun = _POOL_FACTORIES.get(container_type)
    if not factory_fun:
        raise ValueError(f"No factory registered for type: {container_type}")
    return factory_fun(store_instance)
