$$
\operatorname{Node-Lin}^{(l)} =  \sum_{t \in \mathcal{A}} d^{(l-1)}d^{(l)} + d^{(l)}
$$

$$
\operatorname{Rel-Lin}^{(l)} = \sum_{t \in \mathcal{R}} h^{l}(d^{(l)} / h^{l})^2 + d^{(l)} 
$$

$$
\operatorname{HGT}^{(l)} = 4 \cdot \operatorname{Node-Lin}^{(l)} + 3 \cdot \operatorname{Rel-Lin}^{(l)}
$$

$$
\operatorname{Aux-Emb} = d^{(0)} |\mathcal{V}|\cdot \operatorname{embed-ratio}
$$

$$
\operatorname{Att-Lin} = (d^{(L)}d^{(L)} + d^{(L)})h
$$

$$
\operatorname{Feat-Lin} = \sum_{t \in \mathcal{A}} d^{\mathcal{X}(t)}d^{(0)} + d^{(0)}
$$

$$
MGTCOM = \operatorname{Aux-Emb} + \operatorname{Att-Lin} + \operatorname{Feat-Lin} + \sum_{l=1}^L \operatorname{HGT}^{(l)}
$$


