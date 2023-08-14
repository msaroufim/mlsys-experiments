# How to rewrite rotary positional embeddings w/o complex values
# It's going to be slow though but will work on HW where complex numbers are not supported

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    # Real and imaginary parts of xq and xk
    xq_real, xq_imag = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_real, xk_imag = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)

    # Real and imaginary parts of freqs_cis
    freqs_cis_real, freqs_cis_imag = torch.real(freqs_cis), torch.imag(freqs_cis)
    
    # Broadcast freqs_cis to match the shape of xq and xk
    freqs_cis_real = reshape_for_broadcast(freqs_cis_real, xq_real)
    freqs_cis_imag = reshape_for_broadcast(freqs_cis_imag, xq_imag)

    # Apply the rotary position embedding to the query and key
    xq_out = torch.cat([(xq_real * freqs_cis_real - xq_imag * freqs_cis_imag), 
                        (xq_real * freqs_cis_imag + xq_imag * freqs_cis_real)], dim=-1).flatten(3)

    xk_out = torch.cat([(xk_real * freqs_cis_real - xk_imag * freqs_cis_imag), 
                        (xk_real * freqs_cis_imag + xk_imag * freqs_cis_real)], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
