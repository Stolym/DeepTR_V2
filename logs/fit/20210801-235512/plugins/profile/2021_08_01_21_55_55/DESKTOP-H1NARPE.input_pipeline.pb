$	???g2??? <o??z???x?!^h??HK??	!       "X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsyxρ?y?1yxρ?y?r2"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ^h??HK??? 3??O??1??l??p_?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsvk???y?1vk???y?r4"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails???W?x?1???W?x?r5"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails???W?x?1???W?x?r6"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-C??6z?1-C??6z?r7"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??.??y?1??.??y?r8"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails	z???x?1z???x?r9"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails
1]??ax?11]??ax?r10*	43333?T@2E
Iterator::Root?ڊ?e???!7??Mo?X@)??ݓ????1O@t??GO@:Preprocessing2T
Iterator::Root::ParallelMapV2?:pΈҞ?! :_?#-B@)?:pΈҞ?1 :_?#-B@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??H?}M?!???,d??)??H?}M?1???,d??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI/~M~G J@Qс????G@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	T?qs*y???̔?ߒ?!? 3??O??	!       "$	|??[:w???K?8W???l??p_?!-C??6z?*	!       2	!       :	!       B	!       J	!       R	!       Z	!       b	!       JGPUb q/~M~G J@yс????G@