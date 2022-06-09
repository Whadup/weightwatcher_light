import pandas as pd
import torchvision.models as models
from weightwatcher_light import weightwatcher
if __name__ == "__main__":
    results = []
    for model_cls in [models.vgg11, models.vgg13, models.vgg16, models.vgg19]:
        print(f"======{model_cls.__name__}======")
        model = model_cls(pretrained=True).cuda()
        statistics = weightwatcher(model, (1, 3, 32, 32), verbose=False, debug=False)
        print(pd.DataFrame(statistics["layers"]).to_markdown())
        statistics.pop("layers")
        results.append(statistics)
    
    for n, r in zip(["VGG11", "VGG13", "VGG16", "VGG19"], results):
        print(n, r)
    #VGG13: {'spectral_norm': 81.13524187528171, 'alpha': 2.0631472881023702, 'weighted_alpha': 3.737522625643188}
    #VGG16: {'spectral_norm': 65.65331041812897, 'alpha': 2.0158966332674026, 'weighted_alpha': 3.43908260343017}
    #VGG19: {'spectral_norm': 53.92422615854364, 'alpha': 2.144058164797331, 'weighted_alpha': 3.489194736805615}