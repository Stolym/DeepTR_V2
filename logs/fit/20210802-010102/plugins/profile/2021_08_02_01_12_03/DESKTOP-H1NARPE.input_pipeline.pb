$	??K??od@?~???q@r?Pi?̎?!hur???~@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'hur???~@t{Ic=@1???s??{@I??d??j/@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ????y????ù???1??IӠh^?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!r?Pi?̎?V?j-?B??1? 3??O\?r11*	    xy@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??7??d??!???LI@)鷯????1?_aV??G@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?T???N??!`aV??C?@)??ڊ?e??1??bK?m6@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat;?O??n??!???0?!@)?J?4??1kq??} @:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatvq?-??!+?/?@)B>?٬???1???u?@:Preprocessing2T
Iterator::Root::ParallelMapV2Zd;?O???!1?Qġ?@)Zd;?O???11?Qġ?@:Preprocessing2E
Iterator::Root??B?iޡ?!)??I? !@)tF??_??1B2???\@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch??~j?t??!?ɀz?@)??~j?t??1?ɀz?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipmV}??b??!Ɩ???N@)???_vO~?1??Y?"??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice??H?}m?!n?
?E??)??H?}m?1n?
?E??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorF%u?k?!:i$???)F%u?k?1:i$???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangea2U0*?c?!??\w????)a2U0*?c?1??\w????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice??_?LU?!???+?j??)??_?LU?1???+?j??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI ?@=%H"@Q??WX??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	5m??	k#@??Z\?0@V?j-?B??!t{Ic=@	!       "$	p-???b@7Iµ?p@? 3??O\?!???s??{@*	!       2	!       :	*?????@?ԕ??#"@!??d??j/@B	!       J	!       R	!       Z	!       b	!       JGPUb q ?@=%H"@y??WX??V@