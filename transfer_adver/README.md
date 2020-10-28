# TransferingAdversarialAttacks
Research code for the concept of transferring adversarial attacks based on dataset intersections.

Demo with CIFAR100:

```
python run_experiment_ART.py --epoch 25 --attack pgd --adv-steps 50 --num-shared-classes 50 --percent-shared-data 25
```

This code uses IBM's Adversarial Robustness 360 Toolbox (ART)

```
@article{art2018,
    title = {Adversarial Robustness Toolbox v1.0.0},
    author = {Nicolae, Maria-Irina and Sinn, Mathieu and Tran, Minh~Ngoc and Buesser, Beat and Rawat, Ambrish and Wistuba, Martin and Zantedeschi, Valentina and Baracaldo, Nathalie and Chen, Bryant and Ludwig, Heiko and Molloy, Ian and Edwards, Ben},
    journal = {CoRR},
    volume = {1807.01069},
    year = {2018},
    url = {https://arxiv.org/pdf/1807.01069}
}
```
