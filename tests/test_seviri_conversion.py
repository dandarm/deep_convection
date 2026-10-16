from concurrent.futures import ThreadPoolExecutor
import pytest
import emma_gpm.seviri_conversion as conversion


@pytest.mark.parametrize('prefetch',[1,2,4])
def test_parallel_completion_preserves_input_order_and_defers_failures_to_parent(monkeypatch,prefetch):
    submitted=[]
    def executor(**kwargs):
        pool=ThreadPoolExecutor(max_workers=kwargs['max_workers'])
        original=pool.submit
        def submit(fn,*args):
            submitted.append(args[0])
            return original(fn,*args)
        pool.submit=submit
        return pool
    def read(path,bounds):
        if path==2:raise ValueError('buffer is smaller than requested size')
        return path*10
    monkeypatch.setattr(conversion,'ProcessPoolExecutor',executor)
    monkeypatch.setattr(conversion,'_convert',read)
    stream=conversion.ordered_native_conversions(range(9),None,workers=2,prefetch=prefetch)
    path,result=next(stream)
    assert path==0 and result()==0 and submitted==list(range(2*prefetch))
    observed=[path]
    for path,result in stream:
        observed.append(path)
        if path==2:
            with pytest.raises(ValueError,match='buffer is smaller'):result()
        else:assert result()==path*10
    assert observed==list(range(9))


def test_serial_fallback_keeps_validation_and_recovery_in_parent():
    calls=[]
    def load(path,bounds,**kwargs):
        calls.append((path,kwargs['calibration']))
        raise ValueError('native parser failure')
    stream=conversion.ordered_native_conversions(['a'],None,serial_loader=load)
    path,result=next(stream)
    assert path=='a' and not calls
    with pytest.raises(ValueError,match='native parser failure'):result()
    assert calls==[('a','brightness_temperature')]
    stream.close()


@pytest.mark.parametrize('workers,threads',[(-1,2),(1,0)])
def test_invalid_process_limits_rejected(workers,threads):
    with pytest.raises(ValueError):list(conversion.ordered_native_conversions([],None,workers=workers,threads=threads))


def test_invalid_prefetch_rejected():
    with pytest.raises(ValueError):list(conversion.ordered_native_conversions([],None,workers=1,prefetch=0))
