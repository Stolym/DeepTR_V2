$	]?d8^?S@??&?`?c@??.??y?!??E&??s@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??E&??s@?T??#;@1?_ ?q@IK??>??r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails E??@J???ZI+?????1iUMu_?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??.??y?1??.??y?r6"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!??u6䟩?1??IӠh^?Iޏ?/????r11*	??????A@2T
Iterator::Root::ParallelMapV2?I+???!s???N@)?I+???1s???N@:Preprocessing2E
Iterator::Root??W?2ġ?!?x?3^X@)?
F%u??1%+Y?J?A@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??H?}M?!??0?9@)??H?}M?1??0?9@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI@G????!@Q?H??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?jf-,@???? +@!?T??#;@	!       "$	0?709?Q@ȕ P?a@??IӠh^?!?_ ?q@*	!       2	!       :	?j????,?n???!K??>??B	!       J	!       R	!       Z	!       b	!       JGPUb q@G????!@y?H??V@