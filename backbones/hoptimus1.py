import torch.nn as nn
import torch
import timm
from timm.data import resolve_data_config
from timm.layers import SwiGLUPacked
import torchvision

class base(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True, init_values=1e-5, dynamic_img_size=False)
        self.output_token_dim = 16
        in_features = 1536

        self.adaptor = nn.Sequential(
                        nn.Linear(in_features=in_features,
                                out_features=in_features)
                    )

        self.classifier = nn.Sequential(
                    nn.Linear(in_features=in_features*2, #double as we concatenate cls and mean token
                            out_features=num_classes)
                )
        
    def get_transforms(self):
        transforms = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(224),
                                                     torchvision.transforms.Resize(224),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517))
                                            ]
                                            )
        return transforms
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone.forward_features(x) 
        cls_token = emb[:,0]
        mean_token = torch.mean(emb[:, 5:],dim=1) #token 1-4 are registers
        return torch.cat((cls_token,mean_token),dim=1) #concatenates the cls token and mean features of the patch tokens

    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone.forward_features(x) 
        cls_token = emb[:,0:1]
        return torch.cat((cls_token,emb[:, 5:]),dim=1)
    
    def extract_feature_dict(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x) #returns a dict with last_hidden_state(cls + patchtokens) and pooler_output (cls_token)
        cls_token = emb[:,0]
        mean_token = torch.mean(emb[:, 5:],dim=1) #token 1-4 are registers
        return {"embedding":torch.cat((cls_token,mean_token),dim=1), "patch_tokens":emb[:, 5:]}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.extract_features(x)
        out = self.classifier(emb)
        return out

class base_precomputed(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        in_features = 1536
        self.output_token_dim = 16

        self.adaptor = nn.Sequential(
                        nn.Linear(in_features=in_features,
                                out_features=in_features)
                    )

        self.classifier = nn.Sequential(
                    nn.Linear(in_features=in_features*2, #double as we concatenate cls and mean token
                            out_features=num_classes),
                )
        
    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adaptor(x)
        cls_token = x[:,0,:]
        mean_token = torch.mean(x[:,1:,:],dim=1)
        return torch.cat((cls_token,mean_token),dim=1) #concatenates the cls token and mean features of the patch tokens

    def extract_feature_dict(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.adaptor(x)
        cls_token = emb[:,0,:]
        mean_token = torch.mean(emb[:,1:,:],dim=1)
        return {"embedding":torch.cat((cls_token,mean_token),dim=1), "patch_tokens":emb[:,1:,:]}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.extract_features(x)
        out = self.classifier(emb)
        return out

def get_backbone(num_classes, ttt_mode=None,precomputed=None):
    if precomputed:
        if not ttt_mode:
            class hoptimus1(base_precomputed):
                def __init__(self, num_classes):
                    super().__init__(num_classes)
        
        elif ttt_mode == "multitask":
            class hoptimus1(base_precomputed):
                def __init__(self, num_classes):
                    super().__init__(num_classes)
                
                def extract_feature_dict(self, x: torch.Tensor) -> torch.Tensor:
                    emb = self.adaptor(x)
                    cls_token = emb[:,0,:]
                    mean_token = torch.mean(emb[:, 1:,:],dim=1)
                    return {"embedding":torch.cat((cls_token,mean_token),dim=1), "patch_tokens":emb[:,1:,:]}

                def extract_patch_embeddings_postAdaptor(self, x: torch.Tensor) -> torch.Tensor:
                    emb = self.adaptor(x)
                    return emb[:, 1:,:]
                
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    emb = self.extract_features(x)
                    out = self.classifier(emb)
                    return out
    else:    
        if not ttt_mode:
            class hoptimus1(base):
                def __init__(self, num_classes):
                    super().__init__(num_classes)
        
        elif ttt_mode == "multitask":
            class hoptimus1(base):
                def __init__(self, num_classes):
                    super().__init__(num_classes)

                    in_features = self.backbone.norm.normalized_shape[0]

                    self.adaptor = nn.Sequential(
                        nn.Linear(in_features=in_features,
                                out_features=in_features)
                    )

                def extract_features(self, x: torch.Tensor) -> torch.Tensor:
                    emb = self.backbone(x) 
                    emb = self.adaptor(emb)
                    cls_token = emb[:,0]
                    mean_token = torch.mean(emb[:, 5:],dim=1) # token 1-4 are registers
                    return torch.cat((cls_token,mean_token),dim=1)
                
                def extract_feature_dict(self, x: torch.Tensor) -> torch.Tensor:
                    emb = self.backbone(x) 
                    emb = self.adaptor(emb)
                    cls_token = emb[:,0]
                    mean_token = torch.mean(emb[:, 5:],dim=1) # token 1-4 are registers
                    return {"embedding":torch.cat((cls_token,mean_token),dim=1), "patch_tokens":emb[:, 5:]}
                
                def extract_features_preAdaptor(self, x: torch.Tensor) -> torch.Tensor:
                    emb = self.backbone(x) 
                    cls_token = emb[:,0]
                    mean_token = torch.mean(emb[:, 5:],dim=1) # token 1-4 are registers
                    return torch.cat((cls_token,mean_token),dim=1) #concatenates the cls token and mean features of the patch tokens
                
                def extract_patch_embeddings_postAdaptor(self, x: torch.Tensor) -> torch.Tensor:
                    emb = self.backbone(x) 
                    emb = self.adaptor(emb)
                    return emb[:, 5:]
        
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    emb = self.extract_features(x)
                    out = self.classifier(emb)
                    return out
        
        else:
            raise ValueError(f"Unsupported ttt_mode: {ttt_mode}")
        
    return hoptimus1(num_classes)