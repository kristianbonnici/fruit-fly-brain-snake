# Results and limitations

## Selected controller

- Internal efficacy: adapted V96, checkpoint SHA-256 `de381ff7e64e615489c34fe6925173dbd5d116386d45d1787e92c0c328feffe0`.
- Action readout: V104 adapted source decoder, SHA-256 `b2d65045ecd21abc7a896ac6b12b13722ad857c2e17bd09376980050281f3b08`.
- Full policy identity: `9f6419ed938acc187ca2138a9a39da936e58a80b542c1f9da96b515e6cdb48b4`.

## Development evaluation

| Measure | Result |
| --- | --- |
| Games | 32 |
| Mean food | 10.15625 |
| Median food | 10 |
| At least five food | 26 / 32 (81.25%) |
| Termination | 30 collisions, 2 no-food timeouts |

Scores in original order:

```text
9, 8, 12, 9, 10, 0, 13, 13, 18, 19, 16, 3, 6, 6, 17, 18,
4, 14, 11, 10, 13, 17, 2, 5, 7, 20, 3, 14, 6, 13, 6, 3
```

The viewer includes game 4 (10 food, 157 decisions) and game 25 (20 food, 220 decisions), using zero-based game indices. Selection makes them useful examples; it does not make them an unbiased sample of performance.

## Verification and limits

The original capture report records exact agreement between recaptured full-network endpoints and every available original full-state boundary/output vector. All original actions were reproduced by an independent FP64 readout within the stated numerical tolerance. The release validator checks recorded hashes, dimensions, anatomical identities, connection decoding, board transitions and terminal states without running the network.

These are development results from one training lineage, with demonstrations and learned external interfaces. Three independent training fits, 1,500 final games, and broad causal validation were deferred. Earlier registered gates were not all passed. The V107 candidate was not adopted after failing two checks. The selected V96/V104 history also contains failed gates, retained in the [research reports](research/README.md).

The source release has synthetic numerical tests and replay verification. It does not claim that a fresh clone can reproduce the entire original training run: large historical corpora/checkpoints were pruned, and the frozen research drivers bind to those artifacts.
