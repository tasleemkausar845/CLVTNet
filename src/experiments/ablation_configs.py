from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class AblationConfig:
    """Configuration for ablation studies matching Table 2 in the paper."""
    
    use_local_branch: bool = True
    use_global_branch: bool = True
    
    token_reduction_method: Literal['ctr', 'uniform', 'mean'] = 'ctr'
    token_reduction_factor: int = 2
    
    use_se_in_fusion: bool = True
    use_se_in_local: bool = False
    se_reduction_ratio: int = 4
    
    use_conv_module: bool = True
    conv_module_variant: Literal['standard', 'lite', 'heavy'] = 'lite'
    
    backbone: Literal['clvt', 'mobilevit', 'edgevit', 'vit', 'lvt'] = 'clvt'
    
    use_multiband_ssf: bool = False
    
    def get_model_name(self) -> str:
        """Generate model name based on configuration."""
        components = []
        
        if not self.use_local_branch and self.use_global_branch:
            components.append('CLVT-G')
        elif self.use_local_branch and not self.use_global_branch:
            components.append('CLVT-L')
        elif self.use_local_branch and self.use_global_branch:
            components.append('CLVT')
        else:
            components.append('CLVT-NONE')
        
        if self.token_reduction_method != 'ctr':
            components.append(f'TR-{self.token_reduction_method}')
        
        if not self.use_se_in_fusion:
            components.append('noSE')
        
        if not self.use_conv_module:
            components.append('noConv')
        elif self.conv_module_variant != 'lite':
            components.append(f'Conv-{self.conv_module_variant}')
        
        if self.backbone != 'clvt':
            components.append(f'backbone-{self.backbone}')
        
        if self.use_multiband_ssf:
            components.append('multiband')
        
        return '_'.join(components)
    
    def validate(self):
        """Validate configuration."""
        if not self.use_local_branch and not self.use_global_branch:
            raise ValueError("At least one branch must be enabled")
        
        if self.backbone != 'clvt' and \
           (not self.use_local_branch or not self.use_global_branch):
            raise ValueError("Backbone comparison requires full CLVT model")


ABLATION_CONFIGS = {
    'CLVT-G': AblationConfig(
        use_local_branch=False,
        use_global_branch=True,
        use_se_in_fusion=False,
        use_conv_module=False
    ),
    'CLVT': AblationConfig(
        use_local_branch=True,
        use_global_branch=True,
        use_se_in_fusion=False,
        use_conv_module=False
    ),
    'CLVT_uniform': AblationConfig(
        use_local_branch=True,
        use_global_branch=True,
        token_reduction_method='uniform',
        use_se_in_fusion=False,
        use_conv_module=False
    ),
    'CLVT_mean': AblationConfig(
        use_local_branch=True,
        use_global_branch=True,
        token_reduction_method='mean',
        use_se_in_fusion=False,
        use_conv_module=False
    ),
    'CLVT_w_CTR': AblationConfig(
        use_local_branch=True,
        use_global_branch=True,
        token_reduction_method='ctr',
        use_se_in_fusion=False,
        use_conv_module=False
    ),
    'CLVT-G_w_SE': AblationConfig(
        use_local_branch=False,
        use_global_branch=True,
        use_se_in_fusion=True,
        use_conv_module=False
    ),
    'CLVT-G_wo_SE': AblationConfig(
        use_local_branch=False,
        use_global_branch=True,
        use_se_in_fusion=False,
        use_conv_module=False
    ),
    'CLVT_w_SE': AblationConfig(
        use_local_branch=True,
        use_global_branch=True,
        use_se_in_fusion=True,
        use_conv_module=False
    ),
    'CLVT_wo_SE': AblationConfig(
        use_local_branch=True,
        use_global_branch=True,
        use_se_in_fusion=False,
        use_conv_module=False
    ),
    'CLVTNet': AblationConfig(
        use_local_branch=True,
        use_global_branch=True,
        use_se_in_fusion=True,
        use_conv_module=True,
        conv_module_variant='lite'
    )
}


def get_ablation_config(name: str) -> AblationConfig:
    """Get predefined ablation configuration by name."""
    if name in ABLATION_CONFIGS:
        return ABLATION_CONFIGS[name]
    else:
        raise ValueError(f"Unknown ablation config: {name}. "
                        f"Available: {list(ABLATION_CONFIGS.keys())}")