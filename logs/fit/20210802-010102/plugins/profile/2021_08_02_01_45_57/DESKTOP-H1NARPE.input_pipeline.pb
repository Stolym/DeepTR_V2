$	Ey??0?e@???J??r@??m?2??!Q????O?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'Q????O?@?????8@1?Ӻ??}@I=ԶaP4@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ??m?2????? !ʇ?1?'eRC[?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?%?<Y????????1? 3??O\?r11*	23333Sz@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??????!ڵ$??2J@)?	h"lx??1?:??H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?Fx$??!??a?2?=@)??ʡE???1?ـF?G2@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatk?w??#??!??xP'@)EGr????1w-I??,&@:Preprocessing2T
Iterator::Root::ParallelMapV2??JY?8??!d?b???@)??JY?8??1d?b???@:Preprocessing2E
Iterator::Root??ZӼ???!?5
N?_#@)jM????1????~#@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??ݓ????!5t
??@)9??v????1e?????@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?{??Pk??!+[]?@)?{??Pk??1+[]?@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch?j+??݃?!?O?ml@)?j+??݃?1?O?ml@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??ׁsF??!?u??0N@)? ?	??1U)?p????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice??_vOf?!4H?4H???)??_vOf?14H?4H???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangea2U0*?c?!?#?g?;??)a2U0*?c?1?#?g?;??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice?~j?t?X?!?lu????)?~j?t?X?1?lu????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI???y?L!@QK?o?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	;u???? @?????,@??? !ʇ?!?????8@	!       "$	h?ej??c@???5q@?'eRC[?!?Ӻ??}@*	!       2	!       :	[Qp??@??/?t'@!=ԶaP4@B	!       J	!       R	!       Z	!       b	!       JGPUb q???y?L!@yK?o?V@