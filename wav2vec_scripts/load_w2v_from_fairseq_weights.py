
def load_fairseq_weights(model, attribute, state_dict, config):

    getattr(model, attribute).masked_spec_embed.data = state_dict['mask_emb']
    for i in range(config.num_feat_extract_layers):
        getattr(model, attribute).feature_extractor.conv_layers[i].conv.weight.data = state_dict[f'feature_extractor.conv_layers.{i}.0.weight']
        if i == 0:
            getattr(model, attribute).feature_extractor.conv_layers[i].layer_norm.weight.data = state_dict[f'feature_extractor.conv_layers.0.2.weight']
            getattr(model, attribute).feature_extractor.conv_layers[i].layer_norm.bias.data = state_dict[f'feature_extractor.conv_layers.0.2.bias']
    getattr(model, attribute).feature_projection.projection.weight.data = state_dict['post_extract_proj.weight']
    getattr(model, attribute).feature_projection.projection.bias.data = state_dict['post_extract_proj.bias']

    getattr(model, attribute).encoder.pos_conv_embed.conv.weight_g.data = state_dict['encoder.pos_conv.0.weight_g']
    getattr(model, attribute).encoder.pos_conv_embed.conv.weight_v.data = state_dict['encoder.pos_conv.0.weight_v']
    getattr(model, attribute).encoder.pos_conv_embed.conv.bias.data = state_dict['encoder.pos_conv.0.bias']
    for i in range(config.num_hidden_layers):
        getattr(model, attribute).encoder.layers[i].attention.k_proj.weight.data = state_dict[f'encoder.layers.{i}.self_attn.k_proj.weight']
        getattr(model, attribute).encoder.layers[i].attention.k_proj.bias.data = state_dict[f'encoder.layers.{i}.self_attn.k_proj.bias']
        getattr(model, attribute).encoder.layers[i].attention.v_proj.weight.data = state_dict[f'encoder.layers.{i}.self_attn.v_proj.weight']
        getattr(model, attribute).encoder.layers[i].attention.v_proj.bias.data = state_dict[f'encoder.layers.{i}.self_attn.v_proj.bias']
        getattr(model, attribute).encoder.layers[i].attention.q_proj.weight.data = state_dict[f'encoder.layers.{i}.self_attn.q_proj.weight']
        getattr(model, attribute).encoder.layers[i].attention.q_proj.bias.data = state_dict[f'encoder.layers.{i}.self_attn.q_proj.bias']
        getattr(model, attribute).encoder.layers[i].attention.out_proj.weight.data = state_dict[f'encoder.layers.{i}.self_attn.out_proj.weight']
        getattr(model, attribute).encoder.layers[i].attention.out_proj.bias.data = state_dict[f'encoder.layers.{i}.self_attn.out_proj.bias']

        getattr(model, attribute).encoder.layers[i].layer_norm.weight.data = state_dict[f'encoder.layers.{i}.self_attn_layer_norm.weight']
        getattr(model, attribute).encoder.layers[i].layer_norm.bias.data = state_dict[f'encoder.layers.{i}.self_attn_layer_norm.bias']

        getattr(model, attribute).encoder.layers[i].feed_forward.intermediate_dense.weight.data = state_dict[f'encoder.layers.{i}.fc1.weight']
        getattr(model, attribute).encoder.layers[i].feed_forward.intermediate_dense.bias.data = state_dict[f'encoder.layers.{i}.fc1.bias']
        getattr(model, attribute).encoder.layers[i].feed_forward.output_dense.weight.data = state_dict[f'encoder.layers.{i}.fc2.weight']
        getattr(model, attribute).encoder.layers[i].feed_forward.output_dense.bias.data = state_dict[f'encoder.layers.{i}.fc2.bias']

        getattr(model, attribute).encoder.layers[i].final_layer_norm.weight.data = state_dict[f'encoder.layers.{i}.final_layer_norm.weight']
        getattr(model, attribute).encoder.layers[i].final_layer_norm.bias.data = state_dict[f'encoder.layers.{i}.final_layer_norm.bias']

    getattr(model, attribute).encoder.layer_norm.weight.data = state_dict['encoder.layer_norm.weight']
    getattr(model, attribute).encoder.layer_norm.bias.data = state_dict['encoder.layer_norm.bias']
    return model
