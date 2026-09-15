import copy
import unittest
import numpy as np
from snake import Snake
from snake_whole.full_recording_v91 import N,full_moves,verify_part,policy_boundary,trial
from snake_whole.visual_recording_v85 import game_state,image_hash,state_hash


class BoundedFullRecordingTests(unittest.TestCase):
    def test_declared_coverage_full_boundary_copy_and_complete_original_game(self):
        game=Snake(seed=0,size=16);restored=Snake(seed=0,size=16)
        for start in (0,4):
            images=[];hashes=[];states=[]
            for _ in range(4):
                images.append(game.pixels().copy());hashes.append(image_hash(game.pixels()));game.step(1);states.append(state_hash(game))
            a=dict(actions=np.ones(4,np.int32),logits=np.tile([0.,1.,0.],(4,1)),images=np.asarray(images,np.float32),
                image_hashes=np.asarray(hashes),state_hashes=np.asarray(states),graded_states=np.zeros((4,N),np.float32),
                graded_moves=np.arange(start,start+4,dtype=np.int32),boundary_state=np.zeros((1,N),np.float32),
                output_states=np.zeros((4,64),np.float32),output_indices=np.arange(64,dtype=np.int32),
                view_logits=np.zeros((4,8,17),np.float32),raw_perception=np.zeros((4,17),np.float32),
                sensors=np.tile(np.array([0,0,5/12,5/12,5/12],np.float32),(4,1)),sensory_drive=np.zeros((4,256),np.float32),
                neuron_ids=np.arange(N,dtype=np.int64),substep_moves=np.arange(start,start+4,dtype=np.int32),graded_substeps=np.zeros((4,10,N),np.float32))
            meta=dict(format='bounded-full-game-recording-v91',start_move=start,stop_move=start+4,simulated_ms=(start+4)*100,
                substep_ms=10,neuron_count=N,pair_count=25563197,synapse_count=124025046,input_frame_order=['previous','current'],
                view_order=list(range(8)),game_index=0,saved_game=game_state(game),policy_identity='synthetic-format-fixture')
            if start==0:
                boundary=policy_boundary(a,meta);a['boundary_state'][0,100]=.2
                self.assertEqual(boundary['neural_state'][0,100],0);a['boundary_state'][0,100]=0
                for changed in ('output','boundary','coverage','substep'):
                    bad=copy.deepcopy(a)
                    if changed=='output':bad['output_states'][0,0]=.1
                    if changed=='boundary':bad['boundary_state'][0,1]=.1
                    if changed=='coverage':bad['graded_moves'][0]=3
                    if changed=='substep':bad['graded_substeps'][0,-1,1]=.1
                    with self.subTest(changed=changed),self.assertRaises((ValueError,AssertionError)):verify_part(bad,meta,Snake(seed=0,size=16))
                # Explicit output-only coverage retains a complete full boundary.
                sparse=dict(a,graded_moves=np.empty(0,np.int32),graded_states=np.empty((0,N),np.float32),
                    substep_moves=np.empty(0,np.int32),graded_substeps=np.empty((0,10,N),np.float32))
                verify_part(sparse,dict(meta,game_index=2),Snake(seed=0,size=16))
            verify_part(a,meta,restored)
        self.assertEqual(trial(restored,0)['moves'],8)
        np.testing.assert_array_equal(full_moves(1,252,260),np.arange(252,256,dtype=np.int32))
        self.assertEqual(len(full_moves(0,256,320)),0);self.assertEqual(len(full_moves(2,0,64)),0)


if __name__=='__main__':unittest.main()
