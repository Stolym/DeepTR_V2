$	?5??Ef@???7Is@R???T??!??????@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??????@֍wG;@1?E??v~@I??!???3@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails R???T??mU?Y??1$Di?]?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!'??@j???8K?r??1?mO???^?r11*	?????Az@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapw??/???!RKL???J@)?B?i?q??1.?????H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?? ???!?u????@)-??臨?1?^3m 5@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatA??ǘ???!?a?tH#%@)??A?f??1/??g?#@:Preprocessing2T
Iterator::Root::ParallelMapV2?0?*??!h#??G?@)?0?*??1h#??G?@:Preprocessing2E
Iterator::Root?0?*???!?çg##@)n????1h?w??@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch???_vO??!h?F/@)???_vO??1h?F/@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??H?}??!5c?
l@)?HP???1??K?;@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipV-????!"?T??eM@)?ZӼ?}?1?Sъ
??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceǺ???f?!?i?T??)Ǻ???f?1?i?T??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??_?Le?!5+????)??_?Le?15+????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n??b?!h?????)/n??b?1h?????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice??_?LU?!5+????)??_?LU?15+????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI }w??!@Q\??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	E?ɨ2"@??SQI/@mU?Y??!֍wG;@	!       "$	?=XYOd@?????q@$Di?]?!?E??v~@*	!       2	!       :	?`?-??@???	?'@!??!???3@B	!       J	!       R	!       Z	!       b	!       JGPUb q }w??!@y\??V@