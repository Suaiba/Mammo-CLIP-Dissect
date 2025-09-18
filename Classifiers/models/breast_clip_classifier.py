from torch import nn

from breastclip.model.modules import load_image_encoder, LinearClassifier


class BreastClipClassifier(nn.Module):
    def __init__(self, args, ckpt, n_class,finetuned_ckpt=None):
        super(BreastClipClassifier, self).__init__()
        if ckpt is not None:
            self.config = ckpt["config"]["model"]["image_encoder"]
            self.image_encoder = load_image_encoder(ckpt["config"]["model"]["image_encoder"])
            #print(ckpt["config"]["model"]["image_encoder"])
            """
            # Load configuration and weights from the checkpoint
            if finetuned_ckpt is None:
                image_encoder_weights = {}
                for k in ckpt["model"].keys():
                    if k.startswith("image_encoder."):
                        image_encoder_weights[".".join(k.split(".")[1:])] = ckpt["model"][k]
                self.image_encoder.load_state_dict(image_encoder_weights, strict=True)
            else:
                image_encoder_weights = {}
                for k in finetuned_ckpt["model"].keys():
                    if k.startswith("image_encoder."):
                        image_encoder_weights[".".join(k.split(".")[1:])] = ckpt["model"][k]
                self.image_encoder.load_state_dict(image_encoder_weights, strict=True)
                """
            self.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]

        else:
            print("No checkpoint provided, using default efficient net image encoder weights.")
            self.config = {
                "source": "cnn",  # Default source
                "name": args.im_encoder or "tf_efficientnet_b5_ns-detect",  # Default encoder type
                "pretrained": False,  # No pretraining
                "model_type": "cnn",  # Default model type
            }
            self.image_encoder = load_image_encoder(self.config)
            self.image_encoder_type = self.config["model_type"]

        self.arch = args.arch.lower()
        if (
                args.arch.lower() == "upmc_breast_clip_det_b5_period_n_lp" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b5_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_det_b2_period_n_lp" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b2_period_n_lp"):
            print("freezing image encoder to not be trained")
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.classifier = LinearClassifier(feature_dim=self.image_encoder.out_dim, num_class=n_class)
        self.raw_features = None
        self.pool_features = None

    def get_image_encoder_type(self):
        return self.image_encoder_type

    def encode_image(self, image):
        if self.image_encoder_type == "cnn":
            if self.config["name"].lower() == "resnet152" or self.config["name"].lower() == "resnet101":
                image_features = self.image_encoder(image)
                return image_features
            else:
                input_dict = {"image": image, "breast_clip_train_mode": True}
                image_features, raw_features = self.image_encoder(input_dict)
                self.raw_features = raw_features
                self.pool_features = image_features
                return image_features
        else:
            image_features = self.image_encoder(image)
            # get [CLS] token for global representation (only for vision transformer)
            global_features = image_features[:, 0]
            return global_features

    def forward(self, images):
        if self.image_encoder_type.lower() == "swin":
            images = images.squeeze(1).permute(0, 3, 1, 2)
        # get image features and predict
        image_feature = self.encode_image(images)
        logits = self.classifier(image_feature)
        return logits