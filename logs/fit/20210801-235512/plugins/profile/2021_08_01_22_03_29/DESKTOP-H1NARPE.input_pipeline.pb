$	???x?e@6-??\6r@?1 {????!A?9w??@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'A?9w??@?4ӽN?7@1U??-%
}@I????0@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?1 {????C?*q??1rQ-"??[?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?'??7??1Z?rL?_?I9{??/??r11*	33333?@@2E
Iterator::Root8gDio??!I????X@)??ǘ????1?b@H@:Preprocessing2T
Iterator::Root::ParallelMapV2???H??!䝿?i?G@)???H??1䝿?i?G@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??_?LU?!????A#@)??_?LU?1????A#@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI0W2?? @Q?9)??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??5???@??????+@!?4ӽN?7@	!       "$	?~??"\c@?)???p@rQ-"??[?!U??-%
}@*	!       2	!       :	?c7F??@?J?O\?"@!????0@B	!       J	!       R	!       Z	!       b	!       JGPUb q0W2?? @y?9)??V@