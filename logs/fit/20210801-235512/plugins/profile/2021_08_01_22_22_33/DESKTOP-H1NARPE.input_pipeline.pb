$	?(z?c?f@ύխ??s@[?a/???!?ɧ??@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?ɧ??@?"r??@1Ih˹?@I??w?+@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails [?a/????~j?t???1????}rd?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!??D-ͭ??1iUMu_?I3???/??r11*	????̌G@2E
Iterator::Root??&???!??^???X@)J+???1???,J@:Preprocessing2T
Iterator::Root::ParallelMapV2??JY?8??!?k=	G@)??JY?8??1?k=	G@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??H?}M?!?O[h????)??H?}M?1?O[h????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI????|? @Q!?dp?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?h&%%@??B%M2@!?"r??@	!       "$	^??5o?d@?z.???q@iUMu_?!Ih˹?@*	!       2	!       :	?y?S?t@?gE?*?@!??w?+@B	!       J	!       R	!       Z	!       b	!       JGPUb q????|? @y!?dp?V@