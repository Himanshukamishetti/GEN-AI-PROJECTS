import importlib.util
from pathlib import Path

import pytest
from fastapi import HTTPException

MODULE_PATH = Path(__file__).parents[1] / "data-extraction" / "app.py"
spec = importlib.util.spec_from_file_location("extraction_app", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_validate_text_strips_whitespace():
    assert module.validate_text("  invoice text  ") == "invoice text"


def test_validate_text_rejects_blank_input():
    with pytest.raises(HTTPException) as exc:
        module.validate_text("  \n ")
    assert exc.value.status_code == 422


def test_validate_text_rejects_oversized_input():
    with pytest.raises(HTTPException) as exc:
        module.validate_text("x" * (module.MAX_TEXT_CHARS + 1))
    assert exc.value.status_code == 413


def test_schema_defaults_missing_lists_to_empty():
    invoice = module.InvoiceExtraction()
    resume = module.ResumeExtraction()
    assert invoice.line_items == []
    assert resume.skills == []
    assert resume.education == []



# import importlib.util
# from pathlib import Path

# import pytest
# from fastapi import HTTPException

# MODULE_PATH = Path(__file__).parents[1] / "data-extraction" / "app.py"
# spec = importlib.util.spec_from_file_location("extraction_app", MODULE_PATH)
# module = importlib.util.module_from_spec(spec)
# assert spec.loader is not None
# spec.loader.exec_module(module)


# def test_validate_text_strips_whitespace():
#     assert module.validate_text("  invoice text  ") == "invoice text"


# def test_validate_text_rejects_blank_input():
#     with pytest.raises(HTTPException) as exc:
#         module.validate_text("  \n ")
#     assert exc.value.status_code == 422


# def test_validate_text_rejects_oversized_input():
#     with pytest.raises(HTTPException) as exc:
#         module.validate_text("x" * (module.MAX_TEXT_CHARS + 1))
#     assert exc.value.status_code == 413


# def test_schema_defaults_missing_lists_to_empty():
#     invoice = module.InvoiceExtraction()
#     resume = module.ResumeExtraction()
#     assert invoice.line_items == []
#     assert resume.skills == []
#     assert resume.education == []
