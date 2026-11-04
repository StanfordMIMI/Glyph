from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from examples.sbu_cxr import baseline_cohort, to_rgb


def test_prognostic_cohort_excludes_images_after_baseline_and_same_day_deaths():
    clinical = pd.DataFrame([
        dict(to_patient_id='a', visit_start_datetime='1/1/1901', length_of_stay=2,
             visit_concept_name='Inpatient Visit', **{'last.status': 'deceased'}),
        dict(to_patient_id='b', visit_start_datetime='1/1/1901', length_of_stay=1,
             visit_concept_name='Inpatient Visit', **{'last.status': 'deceased'}),
        dict(to_patient_id='c', visit_start_datetime='1/1/1901', length_of_stay=3,
             visit_concept_name='Inpatient Visit', **{'last.status': 'discharged'}),
        dict(to_patient_id='d', visit_start_datetime='1/1/1901', length_of_stay=3,
             visit_concept_name='Inpatient Visit', **{'last.status': 'discharged'}),
    ])
    series = pd.DataFrame([
        dict(PatientID=patient, StudyDate=date, SeriesInstanceUID=str(i),
             BodyPartExamined='CHEST', Modality='CR', SeriesDescription='AP')
        for i, (patient, date) in enumerate([
            ('a', '1901-01-01'), ('a', '1900-12-31'),
            ('b', '1901-01-01'), ('c', '1901-01-02'), ('d', '1901-01-01')])
    ])
    result = baseline_cohort(series, clinical)
    assert result.PatientID.tolist() == ['a', 'd']
    assert result.in_hospital_death.tolist() == [1, 0]
    assert result.image_day_relative_to_admission.tolist() == [-1, 0]
    assert result.PatientID.is_unique
    with pytest.raises(ValueError, match='one row per patient'):
        baseline_cohort(series, pd.concat([clinical, clinical.iloc[:1]]))


def test_radiograph_photometry_and_non_square_padding():
    pixels = np.arange(1, 201, dtype=np.uint16).reshape(10, 20)
    positive = np.asarray(to_rgb(SimpleNamespace(pixel_array=pixels,
        PhotometricInterpretation='MONOCHROME2'), size=20))
    negative = np.asarray(to_rgb(SimpleNamespace(pixel_array=pixels,
        PhotometricInterpretation='MONOCHROME1'), size=20))
    assert positive.shape == (20, 20, 3)
    assert positive[5:15].sum() > 0 and positive[:5].sum() == 0
    np.testing.assert_allclose(positive[5:15].astype(int) + negative[5:15].astype(int),
                               255, atol=1)
