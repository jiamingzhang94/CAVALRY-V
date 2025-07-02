import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientAttention(nn.Module):
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super(EfficientAttention, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels=None, head_count=None, value_channels=None, use_attention=False, dropout_rate=0.0):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = EfficientAttention(out_channels, key_channels, head_count, value_channels)
        
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            self.skip_bn = nn.BatchNorm2d(out_channels)
        else:
            self.skip_conv = nn.Identity()
            self.skip_bn = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        if self.use_attention:
            out = self.attention(out)
            
        residual = self.skip_bn(self.skip_conv(residual))
        out += residual
        return self.activation(out)


class DoubleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels=None, head_count=None, value_channels=None, use_attention=False, dropout_rate=0.0):
        super(DoubleResBlock, self).__init__()
        self.res1 = ResBlock(in_channels, out_channels, key_channels, head_count, value_channels, 
                            use_attention=use_attention, dropout_rate=dropout_rate)
        self.res2 = ResBlock(out_channels, out_channels, key_channels, head_count, value_channels, 
                            use_attention=use_attention, dropout_rate=dropout_rate)
    
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        x = self.dropout(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.up(x)
        x = self.activation(self.bn(self.conv(x)))
        x = self.dropout(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_channels=3, img_size=224, base_channels=64, max_channels=1024, dropout_rate=0.0):
        super(Encoder, self).__init__()
        self.img_channels = img_channels
        self.base_channels = base_channels
        
        self.initial_conv = nn.Conv2d(img_channels, base_channels, kernel_size=7, stride=1, padding=3)
        self.initial_bn = nn.BatchNorm2d(base_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
        self.encoder_blocks = nn.ModuleList()
        
        self.encoder_blocks.append(DoubleResBlock(base_channels, base_channels, 
                                                 use_attention=False, dropout_rate=dropout_rate))
        self.encoder_blocks.append(DownBlock(base_channels, base_channels*2, dropout_rate=dropout_rate))
        
        self.encoder_blocks.append(DoubleResBlock(base_channels*2, base_channels*2, 
                                                 use_attention=False, dropout_rate=dropout_rate))
        self.encoder_blocks.append(DownBlock(base_channels*2, base_channels*4, dropout_rate=dropout_rate))
        
        self.encoder_blocks.append(DoubleResBlock(base_channels*4, base_channels*4, 
                                                 base_channels*2, 8, base_channels*4, 
                                                 use_attention=True, dropout_rate=dropout_rate))
        self.encoder_blocks.append(DownBlock(base_channels*4, base_channels*8, dropout_rate=dropout_rate))
        
        self.encoder_blocks.append(DoubleResBlock(base_channels*8, base_channels*8, 
                                                 base_channels*4, 16, base_channels*8, 
                                                 use_attention=True, dropout_rate=dropout_rate))
        self.encoder_blocks.append(DownBlock(base_channels*8, min(base_channels*16, max_channels), dropout_rate=dropout_rate))
        
        self.bottleneck = DoubleResBlock(min(base_channels*16, max_channels), min(base_channels*16, max_channels), 
                                        base_channels*8, 16, min(base_channels*16, max_channels), 
                                        use_attention=True, dropout_rate=dropout_rate)
        
    def forward(self, x):
        features = []
        
        x = self.activation(self.initial_bn(self.initial_conv(x)))
        
        for i, block in enumerate(self.encoder_blocks):
            if isinstance(block, DoubleResBlock):
                x = block(x)
                features.append(x)
            else:
                x = block(x)
        
        x = self.bottleneck(x)
        
        return x, features


class Decoder(nn.Module):
    def __init__(self, img_channels=3, base_channels=64, max_channels=1024, dropout_rate=0.0):
        super(Decoder, self).__init__()
        
        self.decoder_blocks = nn.ModuleList()
        
        self.decoder_blocks.append(DoubleResBlock(min(base_channels*16, max_channels), min(base_channels*16, max_channels), 
                                                 base_channels*8, 16, min(base_channels*16, max_channels), 
                                                 use_attention=True, dropout_rate=dropout_rate))
        self.decoder_blocks.append(UpBlock(min(base_channels*16, max_channels), base_channels*8, dropout_rate=dropout_rate))
        
        self.decoder_blocks.append(DoubleResBlock(base_channels*8*2, base_channels*8, 
                                                 base_channels*4, 16, base_channels*8, 
                                                 use_attention=True, dropout_rate=dropout_rate))
        self.decoder_blocks.append(UpBlock(base_channels*8, base_channels*4, dropout_rate=dropout_rate))
        
        self.decoder_blocks.append(DoubleResBlock(base_channels*4*2, base_channels*4, 
                                                 base_channels*2, 8, base_channels*4, 
                                                 use_attention=True, dropout_rate=dropout_rate))
        self.decoder_blocks.append(UpBlock(base_channels*4, base_channels*2, dropout_rate=dropout_rate))
        
        self.decoder_blocks.append(DoubleResBlock(base_channels*2*2, base_channels*2, 
                                                 use_attention=False, dropout_rate=dropout_rate))
        self.decoder_blocks.append(UpBlock(base_channels*2, base_channels, dropout_rate=dropout_rate))
        
        self.final_res_block = DoubleResBlock(base_channels, base_channels, 
                                             use_attention=False, dropout_rate=dropout_rate)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels//2, img_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, skip_features=None):
        skip_idx = 0
        
        for i, block in enumerate(self.decoder_blocks):
            if isinstance(block, DoubleResBlock):
                if skip_features is not None and i > 0 and skip_idx < len(skip_features):
                    skip_feature = skip_features[-(skip_idx+1)]
                    
                    if skip_feature.shape[2:] != x.shape[2:]:
                        skip_feature = F.interpolate(skip_feature, size=x.shape[2:], 
                                                   mode='bilinear', align_corners=False)
                    
                    x = torch.cat([x, skip_feature], dim=1)
                    skip_idx += 1
                
                x = block(x)
            else:
                x = block(x)
        
        x = self.final_res_block(x)
        x = self.final_conv(x)
        
        return x

class Generator(nn.Module):
    def __init__(self, img_channels=3, img_size=224, base_channels=64, max_channels=1024, 
                 use_skip_connections=True, dropout_rate=0.0):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.encoder = Encoder(img_channels, img_size, base_channels, max_channels, dropout_rate)
        self.decoder = Decoder(img_channels, base_channels, max_channels, dropout_rate)
        self.use_skip_connections = use_skip_connections
        
    def forward(self, x):
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        features_map, skip_features = self.encoder(x)
        
        if self.use_skip_connections:
            output = self.decoder(features_map, skip_features)
        else:
            output = self.decoder(features_map)

        output = F.interpolate(output, size=(448, 448), mode='bilinear', align_corners=False)
    
        return output
    
    def to_bfloat16(self):
        return self.to(torch.bfloat16)
    
    def to_float16(self):
        return self.to(torch.float16)
    
    def to_float32(self):
        return self.to(torch.float32)

