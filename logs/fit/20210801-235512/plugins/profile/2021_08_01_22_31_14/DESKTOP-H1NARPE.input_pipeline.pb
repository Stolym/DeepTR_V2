$	?k?/h??y1~ x????AA)Z?w?!0/?>:u??	!       "a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ??͎Tߵ?(~??k	??1??!??j?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsvk???y?1vk???y?r4"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails 0/?>:u??+?~NA??1#ظ?]???r5"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails	/PR`Ly?1/PR`Ly?r9"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails
?AA)Z?w?1?AA)Z?w?r10*	?????F@2T
Iterator::Root::ParallelMapV2j?t???!??;?\`H@)j?t???1??;?\`H@:Preprocessing2E
Iterator::Root'???????!{?~VCX@)䃞ͪϕ?1?+?P&H@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??_?LU?!??p.0?@)??_?LU?1??p.0?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 84.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??+?;U@QA?#&/@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?:b????????*??!+?~NA??	!       "$	Ӈ?a??{??VJ<? l???!??j?!#ظ?]???*	!       2	!       :	!       B	!       J	!       R	!       Z	!       b	!       JGPUb q??+?;U@yA?#&/@