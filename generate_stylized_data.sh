cd external/DualStyleGAN

python style_transfer_batch.py \
--style pixar --style_id 9 \
--weight 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  \
--content data/real_space \
--output_path data/style_pixar_9_0705

python style_transfer_batch.py \
--style comic --style_id 34 \
--weight 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7  \
--content data/real_space \
--output_path data/style_comic_34_0707

python style_transfer_batch.py \
--style slamdunk --style_id 66 \
--weight 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 \
--content data/real_space \
--output_path data/style_slamdunk_66_0607

python style_transfer_batch.py \
--style caricature --style_id 17 \
--weight 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 \
--content data/real_space \
--output_path data/style_caricature_17_0808

python style_transfer_batch.py \
--style caricature --style_id 49 \
--weight 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 \
--content data/real_space \
--output_path data/style_caricature_49_0808

python style_transfer_batch.py \
--style caricature --style_id 92 \
--weight 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 \
--content data/real_space \
--output_path data/style_caricature_92_0808

python style_transfer_batch.py \
--style cartoon --style_id 91 \
--weight 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 \
--content data/real_space \
--output_path data/style_cartoon_91_0805

python style_transfer_batch.py \
--style cartoon --style_id 299 \
--weight 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 \
--content data/real_space \
--output_path data/style_cartoon_299_0505

python style_transfer_batch.py \
--style cartoon --style_id 221 \
--weight 0.7 0.7 0.7 0.7 0.7 0.7 0.7 1 1 1 1 1 1 1 1 1 1 1 \
--content data/real_space \
--output_path data/style_cartoon_221_0710

python style_transfer_batch.py \
--style cartoon --style_id 252 \
--weight 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 \
--content data/real_space \
--output_path data/style_cartoon_252_0808
