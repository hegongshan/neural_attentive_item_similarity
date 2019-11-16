### NAIS: Neural Attentive Item Similarity Model for Recommendation
A Non-official Implementation of ”NAIS: Neural Attentive Item Similarity Model for Recommendation”.

If you use the codes for your paper as baseline implementation, please cite the link:

https://github.com/hegongshan/neural_attentive_item_similarity

Official Implemenation (Python 2.7 + TensorFlow 1.x): 
https://github.com/AaronHeee/Neural-Attentive-Item-Similarity-Model

### Requirements

* Python 3

* TensorFlow 2.0+

* NumPy (latest version)

* SciPy (latest version)

### Example to run the codes

* FISM

```
python FISM.py --path data --data_set_name ml-1m --epochs 100 --num_neg 4 --embedding_size 16 --lr 0.01 --alpha 0.0 --regs (1e-7, 1e-7, 1e-7)
```

* NAIS

```
python NAIS.py --pretrain 1 --path data --data_set_name ml-1m --epochs 100 --num_neg 4 --embedding_size 16 --lr 0.01
```

### Experimental Results

* FISM

epochs = 100

|       | HR@10  | NDCG@10 |
| :---: | :----: | :-----: |
| ml-1m | 0.6526 | 0.3857  |

* NAIS

coming soon...

|       | HR@10 | NDCG@10 |
| :---: | :---: | :-----: |
| ml-1m |       |         |

**Last Updated**: November 16, 2019

