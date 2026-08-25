import grpc
import doclens_pb2
import doclens_pb2_grpc
import os 

def test_analyze_document(image_path):
    channel = grpc.insecure_channel('localhost:50051')
    stub = doclens_pb2_grpc.DocumentAnalysisStub(channel)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    request = doclens_pb2.DocumentRequest(
        image_data=image_bytes,
        document_id="test-001",
        filename="Forgery-test_page1.png"
    )

    print("Sending request to gRPC server...")
    response = stub.AnalyzeDocument(request, timeout=300) 

    print("\n--- RESPONSE ---")
    print(f"Document ID: {response.document_id}")
    print(f"Risk Level: {response.risk_level}")
    print(f"Reason: {response.reason}")
    print(f"\nFull Report:\n{response.full_report}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "..", "data", "converted", "Forgery-test_page1.png")
    test_analyze_document(image_path)