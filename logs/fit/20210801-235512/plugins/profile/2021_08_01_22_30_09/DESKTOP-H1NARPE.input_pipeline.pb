$	?|x? ?e@ue?8??r@pD??k???!?m?(?S?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?m?(?S?@8?a?Av9@1????}@I???w?2@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails pD??k???àL??ň?1iUMu_?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?????ə???? !ʗ?1??Z
H?_?r11*	??????B@2T
Iterator::Root::ParallelMapV2tF??_??!]AL? ?O@)tF??_??1]AL? ?O@:Preprocessing2E
Iterator::Rootj?q?????!??Q?وX@)???<,Ԋ?1!&W?kA@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipǺ???F?!W?+????)Ǻ???F?1W?+????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI C?B? @Q??????V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	1?闈? @z???ia-@àL??ň?!8?a?Av9@	!       "$	f??c@???^OBq@iUMu_?!????}@*	!       2	!       :	?8ݟ?@????Q?%@!???w?2@B	!       J	!       R	!       Z	!       b	!       JGPUb q C?B? @y??????V@