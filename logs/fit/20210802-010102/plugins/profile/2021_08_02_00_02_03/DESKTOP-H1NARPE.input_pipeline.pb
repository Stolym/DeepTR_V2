$	?AĝQ@??w?c@?̱????!???a?u@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'???a?u@??)1q<@1M??(??s@I?5??*@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ,am?????
?F?S?1?_??s??r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?̱????1?̱????r4"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!>???@??1!>???@??r5"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!w-!?l??1:?`???d?I/?KR?b??r11*     ?w@)      0=2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapV}??b??!g??r|PJ@)?lV}???1??'hG@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::MapW[??잼?!?W?V??=@)?O??n??1Co{g֮3@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?HP???!???ޙ?#@)???{????1?x	?9!@:Preprocessing2T
Iterator::Root::ParallelMapV2M??St$??!?W?=?@)M??St$??1?W?=?@:Preprocessing2E
Iterator::Root?g??s???!??Bo{g&@)??ׁsF??1?P-R??@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch???{????!?x	?9@)???{????1?x	?9@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatr??????!?[?&??@)X9??v???1??kea@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??ܵ??!x??xO?M@)???_vO~?1?s?H??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeU???N@s?!?.???)U???N@s?1?.???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceHP?s?r?!?5?q??)HP?s?r?1?5?q??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n??b?!?I????)/n??b?1?I????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice/n??R?!?I????)/n??R?1?I????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIP??<#@Q?A]?}?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
		l??3?@?@?mp)@!??)1q<@	!       "$	pe?A?O@???a@:?`???d?!M??(??s@*	!       2	!       :	6?9i??;?bn???!?5??*@B	!       J	!       R	!       Z	!       b	!       JGPUb qP??<#@y?A]?}?V@