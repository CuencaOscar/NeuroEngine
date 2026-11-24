import io


def upload(png_bytes):
    return {"image": (io.BytesIO(png_bytes), "scan.png")}


def test_health(client):
    assert client.get("/health").get_json()["status"] == "ok"


def test_legacy_endpoint_returns_top_class(client, png_bytes):
    res = client.post("/api", data=upload(png_bytes), content_type="multipart/form-data")
    assert res.status_code == 200
    assert res.get_json() == {"Non_Demented": "87.65%"}


def test_v2_returns_all_classes_sorted(client, png_bytes):
    res = client.post("/api/v2/classify", data=upload(png_bytes), content_type="multipart/form-data")
    labels = [p["label"] for p in res.get_json()["predictions"]]
    assert labels[0] == "Non_Demented" and len(labels) == 4


def test_missing_and_invalid_images_are_rejected(client):
    assert client.post("/api", data={}).status_code == 400
    bad = {"image": (io.BytesIO(b"nope"), "x.png")}
    assert client.post("/api", data=bad, content_type="multipart/form-data").status_code == 400


def test_url_from_disallowed_host_is_rejected(client, predictor):
    res = client.post("/api", data={"image": "https://evil.example.com/scan.jpg"})
    assert res.status_code == 400
    assert predictor.calls == 0
