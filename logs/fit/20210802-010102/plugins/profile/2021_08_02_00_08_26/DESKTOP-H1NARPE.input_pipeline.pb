$	??:?e@T???2?r@#ظ?]???!?^???S?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?^???S?@?!S>?9@1??E_A ~@I?X5s;-@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails #ظ?]????}?????1c?J!?K\?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?W?\T??ŭ??ڇ?1rQ-"??[?r11*	333333u@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapHP?s??!???!5K@)?٬?\m??15?x+??H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map5?8EGr??!@???M=@)?b?=y??1?9?&/,@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?{??Pk??!????l.@)tF??_??1
?[??,@:Preprocessing2E
Iterator::Root???{????!??!5?8#@)??ZӼ???1?`??}@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatr??????!?&oe?@)ŏ1w-!??1?????@:Preprocessing2T
Iterator::Root::ParallelMapV2?HP???!?}?	?@)?HP???1?}?	?@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetchg??j+???!X?9?&?@)g??j+???1X?9?&?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??ݓ????!sHM0ފN@) ?o_?y?1?R????:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range????Mbp?!?sHM0???)????Mbp?1?sHM0???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?~j?t?h?!???sHM??)?~j?t?h?1???sHM??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{?G?zd?!萚`????){?G?zd?1萚`????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice/n??R?!???!5???)/n??R?1???!5???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI@?PQ??@Q???
FW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	X???<!@D1T??-@?}?????!?!S>?9@	!       "$	?KC?d@??w*?dq@rQ-"??[?!??E_A ~@*	!       2	!       :	;?Z?|@?HaM?? @!?X5s;-@B	!       J	!       R	!       Z	!       b	!       JGPUb q@?PQ??@y???
FW@