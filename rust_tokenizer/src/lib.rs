use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use std::sync::RwLock;
use tokenizers::Tokenizer;

static TOKENIZER: RwLock<Option<Tokenizer>> = RwLock::new(None);

#[pyfunction]
fn init_tokenizer(path: &str) -> PyResult<bool> {
    let tok = Tokenizer::from_file(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Tokenizer load failed: {e}")))?;
    let mut guard = TOKENIZER.write().unwrap();
    *guard = Some(tok);
    Ok(true)
}

#[pyfunction]
fn encode(py: Python<'_>, text: &str) -> PyResult<PyObject> {
    let guard = TOKENIZER.read().unwrap();
    let tok = guard.as_ref().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Tokenizer not initialized"))?;
    let enc = tok.encode(text, false).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Encode failed: {e}")))?;
    let ids = enc.get_ids();
    let py_list = PyList::new(py, ids.iter());
    Ok(py_list.into())
}

#[pyfunction]
fn decode(_py: Python<'_>, ids: Vec<u32>) -> PyResult<String> {
    let guard = TOKENIZER.read().unwrap();
    let tok = guard.as_ref().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Tokenizer not initialized"))?;
    tok.decode(ids, true).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Decode failed: {e}")))
}

#[pyfunction]
fn special_token_ids(py: Python<'_>) -> PyResult<PyObject> {
    use pyo3::types::PyDict;
    let guard = TOKENIZER.read().unwrap();
    let tok = guard.as_ref().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Tokenizer not initialized"))?;
    let mut map = tok.get_special_tokens(true)
        .into_iter()
        .filter_map(|pt| pt.id.map(|id| (pt.content, id)))
        .collect::<Vec<_>>();
    map.sort_by_key(|(_, id)| *id);
    let dict = PyDict::new(py);
    for (content, id) in map { dict.set_item(PyString::new(py, &content), id)?; }
    Ok(dict.into())
}

#[pymodule]
fn tokenizer_native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(special_token_ids, m)?)?;
    Ok(())
}
